"""Utility functions for interfacing with NCUT."""

import numpy as np
import torch
from ncut_pytorch import rgb_from_cosine_tsne_3d


def get_ncut_eigenvectors(
    affinity: torch.Tensor,
    num_eigenvectors: int = 30,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gets the eigenvectors for NCUT visualization from an affinity matrix that may be
    asymmetric."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    affinity = affinity.to(device)

    # Make the affinity matrix symmetric.
    affinity = affinity + affinity.t()
    affinity /= 2

    sqrt_diagonal = affinity.sum(dim=1).sqrt_()
    affinity /= sqrt_diagonal[:, None]
    affinity /= sqrt_diagonal[None, :]

    eigenvectors, eigenvalues, _ = torch.svd_lowrank(affinity, q=num_eigenvectors)
    eigenvectors = eigenvectors.real
    eigenvalues = eigenvalues.real
    sorted_indices = eigenvalues.argsort(descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]

    # Correct the flipping signs of the eigenvectors.
    eigenvector_signs = eigenvectors.sum(dim=0).sign()
    eigenvectors *= eigenvector_signs

    return eigenvectors


def get_ncut_colors(
    affinity: torch.Tensor,
    num_eigenvectors: int = 30,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Gets the NCUT color of each token from an affinity matrix that may be
    asymmetric."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eigenvectors = get_ncut_eigenvectors(affinity, num_eigenvectors, device)

    _, tsne_rgb = rgb_from_cosine_tsne_3d(eigenvectors, device=device)
    ncut_colors = tsne_rgb.numpy()

    return ncut_colors
