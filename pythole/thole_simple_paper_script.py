from pathlib import Path
import math
import typing as tp

import h5py
import numpy as np
from numpy.typing import NDArray


ANGSTROM_TO_BOHR = 1.8897261258369282
HARTREE_TO_KCALPERMOL = 27.211386024367243 * 1.6021766208e-19 * 6.022140857e23 / 4184


def check_shapes_and_filter_dummy_entries(
    coords: NDArray[np.floating],
    alphas: NDArray[np.floating],
    external_efield: NDArray[np.floating],
) -> tp.Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    r"""This function takes a set of coordinates, polarizabilities and external
    efield vectors, and filters all of those where the coordinates are all
    zero, external efield and polarizabilities are all zero"""
    # Check coorrect shapes of inputs
    assert coords.shape == external_efield.shape
    assert coords.shape == alphas.shape + (3,)

    indices_with_zero_coords = np.argwhere((coords == 0).all((1, 2))).squeeze()
    indices_with_zero_alphas = np.argwhere((alphas == 0).all(-1)).squeeze()
    indices_with_zero_efield = np.argwhere((external_efield == 0).all((1, 2))).squeeze()

    assert (indices_with_zero_alphas == indices_with_zero_coords).all()
    assert (indices_with_zero_alphas == indices_with_zero_efield).all()

    coords = np.delete(coords, indices_with_zero_coords, axis=0)
    alphas = np.delete(alphas, indices_with_zero_coords, axis=0)
    external_efield = np.delete(external_efield, indices_with_zero_coords, axis=0)

    # Check coorrect shapes of outputs inputs
    assert coords.shape == external_efield.shape
    assert coords.shape == alphas.shape + (3,)
    return coords, alphas, external_efield


def thole_energy(
    coords: NDArray[np.floating],
    alphas: NDArray[np.floating],
    external_efield: NDArray[np.floating],
    damp_factor: float,
) -> NDArray[np.floating]:

    # Comment following lines if there are no dummy entries
    coords, alphas, external_efield = check_shapes_and_filter_dummy_entries(
        coords, alphas, external_efield
    )

    thole_pair_matrix = calc_pair_dipole_field_matrix(
        coords=coords, alphas=alphas, damp_factor=damp_factor
    )
    thole_pair_matrix_3a3a = reshape_dipole_field_to_3a3a(thole_pair_matrix)

    inv_alphas_3a3a = repeat_invert_and_reshape_atomic_alphas_to_3a3a(alphas)
    eff_alpha_matrix_3a3a = np.linalg.inv(thole_pair_matrix_3a3a + inv_alphas_3a3a)
    external_efield_3a = external_efield.reshape(
        external_efield.shape[0], 3 * external_efield.shape[1]
    )

    dipoles_3a = np.matmul(
        eff_alpha_matrix_3a3a,
        np.expand_dims(external_efield_3a, -1),
    ).reshape(-1, external_efield_3a.shape[1])
    return -0.5 * (dipoles_3a * external_efield_3a).sum(axis=-1)


def repeat_invert_and_reshape_atomic_alphas_to_3a3a(
    alphas: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""In general atomic polarizability matrices are approximated to be
    isotropic. This creates an array for 1 / alphas of shape 3a x 3a. Each set
    of 3 consecutive values in the diagonal should be equal The first dimension
    is the batch dimension"""
    # Sanity Check
    assert alphas.ndim == 2
    conf_num = alphas.shape[0]
    atoms_num = alphas.shape[1]
    inv_alphas: NDArray[np.floating] = np.repeat((1 / alphas), 3, axis=-1).reshape(
        conf_num, 3 * atoms_num, 1
    )
    inv_alphas = inv_alphas * np.expand_dims(np.eye(3 * atoms_num), 0)
    #  Sanity checks
    if inv_alphas.shape[1] > 2:
        assert inv_alphas.shape == (conf_num, 3 * atoms_num, 3 * atoms_num)
        assert inv_alphas[0, 0, 0] == inv_alphas[0, 1, 1]
        assert inv_alphas[0, 1, 2] == 0
        assert inv_alphas[0, 2, 1] == 0
        assert inv_alphas[0, 1, 1] == inv_alphas[0, 2, 2]
    assert inv_alphas.dtype == np.floating
    return inv_alphas  # type: ignore


def reshape_dipole_field_to_3a3a(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    conf_num = matrix.shape[0]
    atoms_num = matrix.shape[1]
    # The diatomics clearly show no effect in the permutation
    permutation = (0, 1, 3, 2, 4)  # perumte axes 2 and 3
    # permutation = (0, 1, 2, 3, 4)  # perumte axes 2 and 3
    return np.transpose(matrix, permutation).reshape(
        conf_num, atoms_num * 3, atoms_num * 3
    )


# An induced dipole is generated at the location of each atom.
# The electric field due to the induced dipoles can be calculated
# using the dipole field matrix
def calc_pair_dipole_field_matrix(
    coords: NDArray[np.floating],
    alphas: NDArray[np.floating],
    damp_factor: float,
) -> NDArray[np.floating]:
    # TODO: could be improved to N (N - 1) / 2 instead of N^2 if needed
    # Input must be shape C x A x 3 (conformations x atoms x 3)
    #  Output is shape C x 3A x 3A if reshape, else C x A x A x 3 x 3

    # Calculate all pairwise differences
    # Shape is C x A x A x 3
    conf_num = coords.shape[0]
    atoms_num = coords.shape[1]
    if alphas.shape != (conf_num, atoms_num):
        raise ValueError("Incorrect shape for polarizabilities input")

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

    # Calculate the thole damping factors
    # NOTE: self distances are infinite
    pair_alpha_factors = (
        alphas.reshape(conf_num, 1, atoms_num) * alphas.reshape(conf_num, atoms_num, 1)
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


if __name__ == "__main__":
    db = h5py.File(Path.home() / "IO/pythole-data/thole_full_correct.h5")
    for k, conformations in db.items():
        coords = conformations["coordinates"][:]
        # "free" are the ones used in santi's model
        # mbis are the dynamic ones
        alphas = conformations["atomic_polarizabilities_mbis"][:]
        external_efield = conformations["electric_field"][:]

        target_interaction_energy = (
            conformations["pbe.ee_qmmm_energy"][:]
            - conformations["wb97x.mbis_me_qmmm_energy"][:]
        )

        # Convert to Atomic Units
        coords = coords * ANGSTROM_TO_BOHR
        alphas = alphas * ANGSTROM_TO_BOHR**3
        external_efield = external_efield / ANGSTROM_TO_BOHR**2

        pred_interaction_energy = thole_energy(
            coords, alphas, external_efield, damp_factor=0.3
        )
