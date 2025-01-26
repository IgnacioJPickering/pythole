import itertools
from pathlib import Path
import math

from tqdm import tqdm
import h5py
import numpy as np
from numpy.typing import NDArray

ANGSTROM_TO_BOHR = 1.8897261258369282
HARTREE_TO_KCALPERMOL = 27.211386024367243 * 1.6021766208e-19 * 6.022140857e23 / 4184


def thole_energy_kcalpermol(
    coords_ang: NDArray[np.floating],
    alphas_ang3: NDArray[np.floating],
    external_field_eperang2: NDArray[np.floating],
    damp_factor: float,
    epsilon: float,
) -> NDArray[np.floating]:

    if epsilon is math.inf:
        return np.array([0.0] * coords_ang.shape[0], dtype=np.floating)

    coords = coords_ang * ANGSTROM_TO_BOHR
    alphas = alphas_ang3 * ANGSTROM_TO_BOHR**3
    external_field = external_field_eperang2 / ANGSTROM_TO_BOHR**2

    if damp_factor == 0.0:
        energy = (
            -0.5
            * (alphas * (external_field * external_field).sum(-1)).sum(-1)
            / epsilon
        )
    else:
        conf_num = coords.shape[0]
        atoms_num = coords.shape[1]

        thole_pair_matrix = pair_dipole_field_matrix(
            coords=coords,
            alphas=alphas,
            damp_factor=damp_factor,
        )  # 66 %

        # Note that transposition is necessary here
        thole_pair_matrix_3a3a = np.transpose(
            thole_pair_matrix, (0, 1, 3, 2, 4)
        ).reshape(conf_num, atoms_num * 3, atoms_num * 3)
        # Repeat, invert alphas and reshape to 3a3a
        inv_alphas = np.repeat((1 / alphas), 3, axis=-1).reshape(
            conf_num, 3 * atoms_num, 1
        )
        inv_alphas_3a3a = inv_alphas * np.expand_dims(np.eye(3 * atoms_num), 0)

        external_field_3a = external_field.reshape(
            external_field.shape[0], 3 * external_field.shape[1]
        )
        dipoles_3a = np.linalg.solve(
            thole_pair_matrix_3a3a + inv_alphas_3a3a, external_field_3a
        )  # 23 %
        energy = -0.5 * (dipoles_3a * external_field_3a).sum(axis=-1) / epsilon
    return energy * HARTREE_TO_KCALPERMOL


# An induced dipole is generated at the location of each atom.
# The electric field due to the induced dipoles can be calculated
# using the dipole field matrix
def pair_dipole_field_matrix(
    coords: NDArray[np.floating],
    alphas: NDArray[np.floating],
    damp_factor: float,
) -> NDArray[np.floating]:
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

    # Zero self distances to avoid NaN when multiplying 0 * infty
    pair_dist[self_idxs[0], self_idxs[1], self_idxs[2]] = 0.0
    scaled_pair_cbdist = pair_alpha_factors * pair_dist**3
    if damp_factor != 0.0:
        exp_factor = np.exp(-damp_factor * scaled_pair_cbdist)
        pow3_factor = 1 - exp_factor
        pow5_factor = 1 - (1 + damp_factor * scaled_pair_cbdist) * exp_factor

        # Scale the dipole tensor terms
        pow3_term *= pow3_factor.reshape(conf_num, atoms_num, atoms_num, 1, 1)
        pow5_term *= pow5_factor.reshape(conf_num, atoms_num, atoms_num, 1, 1)
    return pow3_term - pow5_term


if __name__ == "__main__":
    disable_tqdm = False
    db = h5py.File(Path.home() / "IO/pythole-data/thole_full_correct.h5")

    # Get target energies for the full dataset
    _target_energies = []
    for k, conformations in db.items():
        pbe_electrostatic_embedding = conformations["pbe.ee_qmmm_energy"][:]
        pbe_distortion_energy = (
            conformations["pbe.qm_energy_sol"][:]
            - conformations["pbe.qm_energy_vac"][:]
        )
        _target_energies.append(pbe_distortion_energy + pbe_electrostatic_embedding)
    target_energies = np.concatenate(_target_energies)
    np.save("./predictions/target.npy", target_energies)

    embed_kinds = ("cm5", "hir", "mbis")
    for embed in embed_kinds:
        _target_energies = []
        for k, conformations in db.items():
            _target_energies.append(conformations[f"wb97x.{embed}_me_qmmm_energy"][:])
        target_energies = np.concatenate(_target_energies)
        np.save(f"./predictions/embed-{embed}.npy", target_energies)

    # Get approximate energies for the full dataset
    damps = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0)
    epsilons = (1000, 50, 20, 10, math.inf)
    alpha_kinds = ("atomtype", "free", "mbis")
    grid = list(itertools.product(damps, epsilons, alpha_kinds))
    for damp, epsilon_str, alpha_kind in tqdm(
        grid, total=len(grid), disable=disable_tqdm
    ):
        epsilon = epsilon_str / 10

        _pred_energies = []
        for k, conformations in db.items():
            correction_kcalpermol = thole_energy_kcalpermol(
                coords_ang=conformations["coordinates"][:],
                alphas_ang3=conformations[f"atomic_polarizabilities_{alpha_kind}"][:],
                external_field_eperang2=conformations["electric_field"][:],
                damp_factor=damp,
                epsilon=epsilon,
            )
            _pred_energies.append(correction_kcalpermol)
        pred_energies = np.concatenate(_pred_energies)
        np.save(
            f"./predictions/pred-{damp}-{str(epsilon_str).zfill(4) if epsilon_str != math.inf else str(9999)}-{alpha_kind}.npy",
            pred_energies,
        )
