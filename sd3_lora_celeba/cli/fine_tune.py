"""The `fine-tune` subcommand."""

import contextlib
import logging
from pathlib import Path
from typing import TypedDict

import click
import torch
import wandb
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch._dynamo.eval_frame import OptimizedModule
from tqdm.auto import tqdm

from ..dataset import ImageAndPromptDataset

_logger = logging.getLogger(__name__)


class FineTuneArgs(TypedDict):
    """Arguments to the `fine-tune` subcommand."""

    dataset_dir: Path
    empty_prompt_dir: Path
    model_name: str
    model_revision: str
    precision: str
    float32_matmul_precision: str
    amp: bool
    device: str | None
    compile_model: bool
    lora_layers: str | None
    lora_blocks: str | None
    lora_rank: int
    lora_alpha: int | None
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_eps: float
    num_training_steps: int
    gradient_accumulation_steps: int
    batch_size: int
    dataloader_num_workers: int
    lr_scheduler: str
    lr_warmup_steps: int
    lr_num_cycles: int
    lr_power: float
    wandb_project: str
    weighting_scheme: str
    logit_mean: float
    logit_std: float
    mode_scale: float
    max_grad_norm: float
    checkpoint_dir: Path
    checkpointing_steps: int


@click.command()
@click.argument(
    "dataset_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument(
    "empty_prompt_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.option("--model-name", default="stabilityai/stable-diffusion-3-medium-diffusers")
@click.option("--model-revision", default="main")
@click.option(
    "--precision",
    type=click.Choice(["float32", "float16", "bfloat16"]),
    default="float32",
)
@click.option(
    "--float32-matmul-precision",
    type=click.Choice(["highest", "high", "medium"]),
    default="highest",
)
@click.option("--amp/--no-amp", default=False)
@click.option("--device", default=None)
@click.option("--compile-model/--no-compile-model", default=False)
@click.option("--lora-layers", default=None)
@click.option("--lora-blocks", default=None)
@click.option("--lora-rank", default=64)
@click.option("--lora-alpha", type=int, default=None)
@click.option("--learning-rate", default=1e-4)
@click.option("--adam-beta1", default=0.9)
@click.option("--adam-beta2", default=0.999)
@click.option("--adam-weight-decay", default=1e-4)
@click.option("--adam-eps", default=1e-8)
@click.option("--num-training-steps", default=1000)
@click.option("--gradient-accumulation-steps", default=1)
@click.option("--batch-size", default=1)
@click.option("--dataloader-num-workers", default=1)
@click.option(
    "--lr-scheduler",
    type=click.Choice(
        [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]
    ),
    default="constant",
)
@click.option("--lr-warmup-steps", default=500)
@click.option("--lr-num-cycles", default=1)
@click.option("--lr-power", default=1.0)
@click.option("--wandb-project", default="sd3-lora")
@click.option(
    "--weighting-scheme",
    type=click.Choice(["sigma_sqrt", "logit_normal", "mode", "cosmap"]),
    default="logit_normal",
)
@click.option("--logit_mean", default=0.0)
@click.option("--logit_std", default=1.0)
@click.option("--mode-scale", default=1.29)
@click.option("--max-grad-norm", default=1.0)
@click.option(
    "--checkpoint-dir",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    default="checkpoints",
)
@click.option("--checkpointing-steps", default=100)
def fine_tune(**kwargs: FineTuneArgs) -> None:
    """Fine-tune a Stable Diffusion 3 model using LoRA."""

    _logger.info("Arguments to fine-tune: %s", kwargs)
    FineTune(kwargs).run()


class FineTune:
    """Implementation of the `fine-tune` subcommand."""

    def __init__(self, args: FineTuneArgs) -> None:
        self.args = args

        self.weight_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args["precision"]]
        self.device: torch.device | None = None
        self.grad_scalar: torch.amp.GradScaler | None = None
        self.transformer: SD3Transformer2DModel | OptimizedModule | None = None
        self.noise_scheduler: FlowMatchEulerDiscreteScheduler | None = None
        self.lora_params: list[torch.nn.Parameter] | None = None
        self.optimizer: torch.optim.AdamW | None = None
        self.dataloader: torch.utils.data.DataLoader | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    def run(self) -> None:
        """Run the subcommand."""

        self.infer_device()

        if self.device.type == "cuda":
            _logger.info(
                "Setting float32_matmul_precision to '%s'",
                self.args["float32_matmul_precision"],
            )
            torch.set_float32_matmul_precision(self.args["float32_matmul_precision"])
        else:
            _logger.info(
                "Ignoring float32_matmul_precision because device '%s' is not CUDA",
                self.device,
            )

        if self.args["amp"]:
            self.grad_scalar = torch.amp.GradScaler(self.device.type)

        self.load_model()
        self.add_lora_adapter()

        self.lora_params = []
        for param in self.transformer.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
                self.lora_params.append(param)

        self.transformer.train()
        if self.args["compile_model"]:
            _logger.info("Compiling the transformer")
            self.transformer = torch.compile(
                self.transformer, fullgraph=True, dynamic=False, mode="max-autotune"
            )

        self.create_optimizer()
        self.create_dataloader()
        self.create_lr_scheduler()

        wandb.login()
        wandb.init(project=self.args["wandb_project"], config=self.args)

        global_step = 0
        progress = tqdm(desc="Training", total=self.args["num_training_steps"])

        for batch_index, examples in enumerate(self.dataloader):
            sync_gradients = (batch_index + 1) % self.args[
                "gradient_accumulation_steps"
            ] == 0
            loss = self.training_step(examples, sync_gradients)

            logs = {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
            progress.set_postfix(logs)

            if sync_gradients:
                wandb.log(logs)
                progress.update()

                global_step += 1
                if global_step % self.args["checkpointing_steps"] == 0:
                    self.save_checkpoint(
                        self.args["checkpoint_dir"] / f"checkpoint-{global_step}"
                    )

        self.transformer = self.unwrap_transformer()
        self.transformer.to(self.weight_dtype)
        self.save_checkpoint(self.args["checkpoint_dir"])

    def infer_device(self) -> None:
        """Infer the PyTorch device to use."""

        device_str = self.args["device"]
        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        _logger.info("Using device: %s", self.device)

    def load_model(self) -> None:
        """Load the transformer and noise scheduler."""

        _logger.info(
            "Loading model '%s', revision '%s'",
            self.args["model_name"],
            self.args["model_revision"],
        )

        self.transformer = (
            SD3Transformer2DModel.from_pretrained(
                self.args["model_name"],
                revision=self.args["model_revision"],
                subfolder="transformer",
            )
            .requires_grad_(False)
            .to(self.device, self.weight_dtype)
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.args["model_name"],
            revision=self.args["model_revision"],
            subfolder="scheduler",
        )

    def add_lora_adapter(self) -> None:
        """Add the LoRA adapter to the transformer."""

        if self.args["lora_layers"] is not None:
            target_modules = [
                layer.strip() for layer in self.args["lora_layers"].split(",")
            ]
        else:
            target_modules = [
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ]

        if self.args["lora_blocks"] is not None:
            lora_blocks = [
                int(block.strip()) for block in self.args["lora_blocks"].split(",")
            ]
            target_modules = [
                f"transformer_blocks.{block}.{module}"
                for block in lora_blocks
                for module in target_modules
            ]

        _logger.info("LoRA target modules: %s", target_modules)

        lora_rank = self.args["lora_rank"]
        lora_alpha = self.args["lora_alpha"]
        if lora_alpha is None:
            lora_alpha = lora_rank
        _logger.info("LoRA rank: %d, alpha: %d", lora_rank, lora_alpha)

        self.transformer.add_adapter(
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
        )

    def create_optimizer(self) -> None:
        """Create the optimizer for fine-tuning."""

        self.optimizer = torch.optim.AdamW(
            self.lora_params,
            lr=self.args["learning_rate"],
            betas=(self.args["adam_beta1"], self.args["adam_beta2"]),
            weight_decay=self.args["adam_weight_decay"],
            eps=self.args["adam_eps"],
        )

    def create_dataloader(self) -> None:
        """Create the dataloader for fine-tuning."""

        dataset = ImageAndPromptDataset(
            self.args["dataset_dir"], self.args["empty_prompt_dir"], self.weight_dtype
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=dataset.get_sampling_weights(),
            num_samples=(
                self.args["num_training_steps"]
                * self.args["gradient_accumulation_steps"]
                * self.args["batch_size"]
            ),
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            sampler=sampler,
            num_workers=self.args["dataloader_num_workers"],
            pin_memory=self.device.type == "cuda",
        )

    def create_lr_scheduler(self) -> None:
        """Create the learning rate scheduler."""

        self.lr_scheduler = get_scheduler(
            self.args["lr_scheduler"],
            self.optimizer,
            num_warmup_steps=self.args["lr_warmup_steps"],
            num_training_steps=self.args["num_training_steps"],
            num_cycles=self.args["lr_num_cycles"],
            power=self.args["lr_power"],
        )

    def training_step(
        self, examples: ImageAndPromptDataset.ExampleBatch, sync_gradients: bool
    ) -> float:
        """Perform a single training step."""

        with (
            torch.amp.autocast(device_type=self.device.type, dtype=self.weight_dtype)
            if self.args["amp"]
            else contextlib.nullcontext()
        ):
            loss = self.compute_loss(examples)
            loss_value = loss.item()
            loss = loss / self.args["gradient_accumulation_steps"]

        if self.args["amp"]:
            loss = self.grad_scalar.scale(loss)
        loss.backward()

        if sync_gradients:
            if self.args["amp"]:
                self.grad_scalar.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.lora_params, self.args["max_grad_norm"])

            if self.args["amp"]:
                self.grad_scalar.step(self.optimizer)
                self.grad_scalar.update()
            else:
                self.optimizer.step()

            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss_value

    def compute_loss(
        self, examples: ImageAndPromptDataset.ExampleBatch
    ) -> torch.Tensor:
        """Compute the loss for a batch of examples."""

        model_input = examples["model_input"].to(self.device)
        noise = torch.randn_like(model_input)
        timesteps, sigmas = self.sample_timesteps_and_sigmas(
            model_input.size(0), model_input.dim()
        )

        noisy_model_input = (1 - sigmas) * model_input + sigmas * noise
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=examples["prompt_embeds"].to(self.device),
            pooled_projections=examples["pooled_prompt_embeds"].to(self.device),
        ).sample
        model_pred = noisy_model_input - sigmas * model_pred

        weighting = compute_loss_weighting_for_sd3(
            self.args["weighting_scheme"], sigmas
        )
        loss = (
            (weighting.float() * (model_pred.float() - model_input.float()).square())
            .flatten(start_dim=1)
            .mean(dim=1)
            .mean()
        )
        return loss

    def sample_timesteps_and_sigmas(
        self, batch_size: int, n_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of timesteps for training, and get the corresponding sigma
        values for scaling the noise."""

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.args["weighting_scheme"],
            batch_size=batch_size,
            logit_mean=self.args["logit_mean"],
            logit_std=self.args["logit_std"],
            mode_scale=self.args["mode_scale"],
        )
        indices = (u * self.noise_scheduler.config["num_train_timesteps"]).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(self.device)

        sigmas = self.noise_scheduler.sigmas[indices].to(self.device, self.weight_dtype)
        while sigmas.dim() < n_dim:
            sigmas = sigmas.unsqueeze(-1)

        return timesteps, sigmas

    def save_checkpoint(self, save_dir: Path) -> None:
        """Save a checkpoint of the LoRA weights."""

        _logger.info("Saving checkpoint to '%s'", save_dir)
        self.args["checkpoint_dir"].mkdir(exist_ok=True)
        StableDiffusion3Pipeline.save_lora_weights(
            save_dir, get_peft_model_state_dict(self.unwrap_transformer())
        )

    def unwrap_transformer(self) -> SD3Transformer2DModel:
        """If the transformer is an OptimizedModule, unwrap it to get the original
        module."""

        if isinstance(self.transformer, OptimizedModule):
            return self.transformer._orig_mod
        return self.transformer
