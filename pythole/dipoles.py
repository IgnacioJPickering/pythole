from enum import Enum
import typing as tp
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class DipoleKind(Enum):
    THOLE = "thole"
    WARSHEL = "warshel"
    APPLEQUIST = "apple"


class MolecularAlphaKind(Enum):
    THOLE = "thole"
    WARSHEL = "warshel"
    APPLEQUIST = "apple"


class EfieldKind(Enum):
    EXTERNAL = "external"
    INDUCED = "induced"
    FULL = "full"


@dataclass
class TholeDampingArgs:
    alphas: NDArray[np.floating]  # Shape should be (C x A)
    damp_factor: float = 1.0


# An induced dipole is generated at the location of each atom.
# The electric field due to the induced dipoles can be calculated
# using the dipole field matrix
def calc_pair_dipole_field_matrix(
    coords: NDArray[np.floating],
    thole_damping_args: tp.Optional[TholeDampingArgs] = None,
) -> NDArray[np.floating]:
    # TODO: could be improved to N (N - 1) / 2 instead of N^2 if needed
    # Input must be shape C x A x 3 (conformations x atoms x 3)
    #  Output is shape C x 3A x 3A if reshape, else C x A x A x 3 x 3

    # Calculate all pairwise differences
    # Shape is C x A x A x 3
    conf_num = coords.shape[0]
    atoms_num = coords.shape[1]
    pair_delta_coords = coords.reshape(conf_num, atoms_num, 1, 3) - coords.reshape(
        conf_num, 1, atoms_num, 3
    )
    # Get self indices
    self_idxs = np.repeat(
        np.eye(atoms_num, dtype=np.int64).reshape(1, atoms_num, atoms_num),
        conf_num,
        axis=0,
    )
    self_idxs = np.argwhere(self_idxs).transpose()
    # Calculate all pairwise distances
    # Shape is C x A x A
    pair_dist = np.linalg.norm(pair_delta_coords, axis=-1)
    # Avoid zero division
    pair_dist[self_idxs[0], self_idxs[1], self_idxs[2]] = math.inf
    pair_inv_dist = 1 / pair_dist
    # Reshape allows elementwise matrix division
    pair_inv_dist = pair_inv_dist.reshape(conf_num, atoms_num, atoms_num, 1, 1)
    # Fast sanity checks
    if pair_inv_dist.shape[0] > 4 and pair_inv_dist.shape[1] > 3:
        assert pair_inv_dist[0, 1, 1, 0, 0] == 0.0
        assert pair_inv_dist[0, 2, 2, 0, 0] == 0.0
        assert pair_inv_dist[4, 3, 3, 0, 0] == 0.0
        assert pair_inv_dist[4, 2, 3, 0, 0] != 0.0
        assert pair_inv_dist[4, 3, 2, 0, 0] == pair_inv_dist[4, 3, 2, 0, 0]

    # Calculate all pairwise "outer products" (Rij)
    # Shape is C x A x A x 3 x 3
    # This generates the matrix:
    #  [[x^2, xy, xz]
    #   [yx, y^2, yz]
    #   [zx, zy, z^2]]
    # In the last two dimensions
    pair_outer_rr = pair_delta_coords.reshape(
        conf_num, atoms_num, atoms_num, 1, 3
    ) * pair_delta_coords.reshape(conf_num, atoms_num, atoms_num, 3, 1)

    # The dipole field matrix has the identity matrix in the final two indices
    identity = np.eye(3).reshape(1, 1, 1, 3, 3)

    #  Caclulate all pairwise dipole field matrices
    #  Tij = (1/rij)^3 * I - 3 (1/rij)^5 * Rij
    # Applequist model
    pow3_term = (pair_inv_dist**3) * identity
    pow5_term = 3 * (pair_inv_dist**5) * pair_outer_rr

    # Calculate the thole damping factors if needed
    if thole_damping_args is not None:
        alphas = thole_damping_args.alphas
        if alphas.shape != (conf_num, atoms_num):
            raise ValueError("Incorrect shape for polarizabilities input")
        damp_factor = thole_damping_args.damp_factor
        # NOTE: self distances are infinite
        pair_alpha_factors = (
            alphas.reshape(conf_num, 1, atoms_num)
            * alphas.reshape(conf_num, atoms_num, 1)
        ) ** (-1 / 2)

        # Sanity checks
        assert not np.isinf(pair_alpha_factors).any()
        assert not np.isnan(pair_alpha_factors).any()

        # Zero self distances to avoid NaN when multiplying 0 * infty
        pair_dist[self_idxs[0], self_idxs[1], self_idxs[2]] = 0.0
        scaled_pair_cbdist = pair_alpha_factors * pair_dist**3
        exp_factor = np.exp(-damp_factor * scaled_pair_cbdist)
        pow3_factor = 1 - exp_factor
        pow5_factor = 1 - (1 + damp_factor * scaled_pair_cbdist) * exp_factor

        # Scale the dipole tensor terms
        pow3_term *= pow3_factor.reshape(conf_num, atoms_num, atoms_num, 1, 1)
        pow5_term *= pow5_factor.reshape(conf_num, atoms_num, atoms_num, 1, 1)
    return pow3_term - pow5_term


def calc_dipoles(
    eff_alpha_matrix: NDArray[np.floating], external_efield: NDArray[np.floating]
) -> NDArray[np.floating]:
    return np.matmul(
        eff_alpha_matrix,
        np.expand_dims(external_efield, -1),
    ).reshape(external_efield.shape[0], -1, 3)


def calc_energy(
    eff_alpha_matrix: NDArray[np.floating], external_efield: NDArray[np.floating]
) -> NDArray[np.floating]:
    # .reshape(external_efield.shape[0], -1, 3)

    dipoles = np.matmul(
        eff_alpha_matrix,
        np.expand_dims(external_efield, -1),
    ).reshape(-1, external_efield.shape[1])
    return -0.5 * (dipoles * external_efield).sum(axis=-1)
