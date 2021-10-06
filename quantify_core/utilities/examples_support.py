# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Utilities used for creating examples for docs/tutorials/tests."""
# pylint: disable=too-many-arguments
from __future__ import annotations

from typing import Union
import numpy as np


def mk_iq_shots(
    n_shots: int,
    sigmas: Union[tuple, list, np.ndarray],
    centers: Union[tuple, list, np.ndarray],
    probabilities: Union[tuple, list, np.ndarray],
    seed: Union[int, None] = 112233,
):
    """
    Generates clusters of (I + 1j*Q) points with a Gaussian distribution with the
    specified sigmas and centers according to the probabilities of each cluster

    .. admonition:: Examples
        :class: dropdown

        .. include:: ./examples/utilities.examples_support.mk_iq_shots.rst.txt

    Parameters
    ----------
    n_shots
        The number of shot to generate.
    sigma
        The sigma of the Gaussian distribution used for both real and imaginary parts.
    centers
        The center of each cluster on the imaginary plane.
    probabilities
        The probabilities of each cluster being randomly selected for each shot.
    seed
        Random number generator passed to ``numpy.random.default_rng``.
    """
    if not (len(sigmas) == len(centers) == len(probabilities)):
        raise ValueError(
            "Incorrect input. sigmas={sigmas}, centers={centers} and "
            f"probabilities={probabilities} must have the same length."
        )

    rng = np.random.default_rng(seed=seed)

    cluster_indices = tuple(range(len(centers)))
    choices = rng.choice(a=cluster_indices, size=n_shots, p=probabilities)

    shots = []
    for idx in cluster_indices:
        num_shots_this_cluster = np.sum(choices == idx)
        i_data = rng.normal(
            loc=centers[idx].real,
            scale=sigmas[idx],
            size=num_shots_this_cluster,
        )
        q_data = rng.normal(
            loc=centers[idx].imag,
            scale=sigmas[idx],
            size=num_shots_this_cluster,
        )
        shots.append(i_data + 1j * q_data)
    return np.concatenate(shots)
