from pathlib import Path
from numpy.typing import NDArray
import numpy as np

from dipoles import DipoleKind, MolecularAlphaKind, EfieldKind


def write_molecular_alpha_file(
    dir_: Path,
    coords: NDArray[np.float64],
    molecular_alpha_evectors: NDArray[np.float64],
    molecular_alpha_evalues: NDArray[np.float64],
    kind: MolecularAlphaKind,
) -> None:
    assert coords.shape[-1] == 3 and coords.ndim == 2
    assert molecular_alpha_evectors.shape == (3, 3)
    assert molecular_alpha_evalues.shape == (3,)
    # Column molecular_alpha[k, :, j] has the normalized eigenvector j, of
    # conformation k

    molecular_alpha = molecular_alpha_evectors * molecular_alpha_evalues.reshape(1, 3)
    #  The start of the eigenvectors is taken to be the center of geometry of the
    #  molecule
    cog = np.mean(coords, 0)  # center of geometry
    with open(
        dir_ / f"molecular-alpha.{kind.value}.csv", mode="wt", encoding="utf-8"
    ) as f:
        f.write("start_x,start_y,start_z,end_x,end_y,end_z\n")
        # Column molecular_alpha[k, :, j] has the eigenvector j, of
        # conformation k so since we want to iterate over the columns, we
        # transpose the array
        for a in molecular_alpha.T:
            #  Displace to center of geometry
            a += cog
            f.write(f"{cog[0]},{cog[1]},{cog[2]},{a[0]},{a[1]},{a[2]}\n")


def write_atomic_alphas_file(
    dir_: Path,
    coords: NDArray[np.float64],
    alphas: NDArray[np.float64],
) -> None:
    assert coords.shape[-1] == 3 and coords.ndim == 2
    assert alphas.shape == coords.shape[:-1]
    with open(dir_ / "atomic-alphas.csv", mode="wt", encoding="utf-8") as f:
        f.write("x,y,z,alpha\n")
        for c, a in zip(coords, alphas):
            f.write(f"{c[0]},{c[1]},{c[2]},{a}\n")  # type: ignore


def write_dipole_file(
    dir_: Path,
    coords: NDArray[np.float64],
    dipoles: NDArray[np.float64],
    kind: DipoleKind,
) -> None:
    assert coords.shape[-1] == 3 and coords.ndim == 2
    assert coords.shape == dipoles.shape
    # Coordinates should be an array of size a x 3
    # Dipoles should be an array of size a x 3
    with open(dir_ / f"dipoles.{kind.value}.csv", mode="wt", encoding="utf-8") as f:
        f.write("start_x,start_y,start_z,end_x,end_y,end_z\n")
        for c, d in zip(coords, dipoles):
            d += c
            f.write(f"{c[0]},{c[1]},{c[2]},{d[0]},{d[1]},{d[2]}\n")  # type: ignore


def write_efield_file(
    dir_: Path,
    coords: NDArray[np.float64],
    efield: NDArray[np.float64],
    kind: EfieldKind,
) -> None:
    # Coordinates should be an array of size a x 3
    # Efield should be an array of size a x 3
    with open(dir_ / f"efield.{kind.value}.csv", mode="wt", encoding="utf-8") as f:
        f.write("start_x,start_y,start_z,end_x,end_y,end_z\n")
        for c, e in zip(coords, efield):
            e += c
            f.write(f"{c[0]},{c[1]},{c[2]},{e[0]},{e[1]},{e[2]}\n")  # type: ignore


def write_xyz_file(
    dir_: Path,
    coords: NDArray[np.float64],
    znums: NDArray[np.int64],
) -> None:
    # Coordinates should be an array of size a x 3
    # Dipoles should be an array of size a x 3
    with open(dir_ / "structure.xyz", mode="wt", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n\n")
        for c, z in zip(coords, znums):
            f.write(f"{z} {c[0]} {c[1]} {c[2]}\n")  # type: ignore
