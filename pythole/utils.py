from dataclasses import dataclass
import typing as tp

import numpy as np
from numpy.typing import NDArray


@dataclass
class HomoEfield:
    x: float
    y: float
    z: float


def repeat_invert_and_reshape_atomic_alphas_to_3a3a(
    alphas: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""In general atomic polarizability matrices are approximated to be
    isotropic. This creates an array for 1 / alphas of shape 3a x 3a. Each set
    of 3 consecutive values in the diagonal should be equal The first dimension
    is the batch dimension"""
    # Sanity Check
    assert alphas.ndim == 2
    conf_num = alphas.shape[0]
    atoms_num = alphas.shape[1]
    inv_alphas: NDArray[np.float64] = np.repeat((1 / alphas), 3, axis=-1).reshape(
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
    assert inv_alphas.dtype == np.float64
    return inv_alphas  # type: ignore


# TODO: Make sure that this permutation is actually correct, very important!!
# I'm 97% sure the transposition is crucial
def reshape_dipole_field_to_3a3a(matrix: NDArray[tp.Any]) -> NDArray[tp.Any]:
    conf_num = matrix.shape[0]
    atoms_num = matrix.shape[1]
    # The diatomics clearly show no effect in the permutation
    permutation = (0, 1, 3, 2, 4)  # perumte axes 2 and 3
    # permutation = (0, 1, 2, 3, 4)  # perumte axes 2 and 3
    return np.transpose(matrix, permutation).reshape(
        conf_num, atoms_num * 3, atoms_num * 3
    )


def reduce_eff_alpha_3a3a_to_molecular_alpha_3x3(
    matrix: NDArray[tp.Any],
) -> NDArray[tp.Any]:
    conf_num = matrix.shape[0]
    matrix = matrix.reshape(
        conf_num, matrix.shape[-2] // 3, 3, matrix.shape[-1] // 3, 3
    )
    matrix = np.sum(matrix, axis=(1, 3))
    assert matrix.shape == (conf_num, 3, 3)
    return matrix


def reshape_efield_to_3a(
    field: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Reshape electric field into a c x 3a array"""
    return field.reshape(field.shape[0], 3 * field.shape[1])


def make_homo_efield(
    coords: NDArray[np.float64], homo_efield: HomoEfield
) -> NDArray[np.float64]:
    r"""
    Generate a homogeneous electric field over all the atoms
    in a set of coordinates
    """
    x = homo_efield.x
    y = homo_efield.y
    z = homo_efield.z
    return np.repeat(
        np.repeat(np.array([[[x, y, z]]], dtype=np.float64), coords.shape[0], axis=0),
        coords.shape[1],
        axis=1,
    )


def repeat_and_reshape_atomic_alphas_to_3a(
    alphas: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""reshape field into a c x 3a array"""
    return np.repeat(alphas, 3, axis=-1)


def check_shapes_and_filter_dummy_entries(
    coords: NDArray[np.float64],
    alphas: NDArray[np.float64],
    external_efield: NDArray[np.float64],
) -> tp.Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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


def znums_from_alphas(alphas: NDArray[np.float64]) -> NDArray[np.int64]:
    r"""Recover atomic numbers from the polarizabilities"""
    factor = 0.14818471
    znums_h: NDArray[np.int64] = np.where(
        np.abs(alphas - (factor * 4.50711)) < 1e-10, 1, 0
    ).astype(np.int64)
    znums_c: NDArray[np.int64] = np.where(
        np.abs(alphas - (factor * 11.3)) < 1e-10, 6, 0
    ).astype(np.int64)
    znums_n: NDArray[np.int64] = np.where(
        np.abs(alphas - (factor * 7.4)) < 1e-10, 7, 0
    ).astype(np.int64)
    znums_o: NDArray[np.int64] = np.where(
        np.abs(alphas - (factor * 5.3)) < 1e-10, 8, 0
    ).astype(np.int64)
    total = (
        (znums_h != 0).astype(np.int64)
        + (znums_c != 0).astype(np.int64)
        + (znums_n != 0).astype(np.int64)
        + (znums_o != 0).astype(np.int64)
    )
    assert (total == 1).all()
    return znums_h + znums_c + znums_n + znums_o
