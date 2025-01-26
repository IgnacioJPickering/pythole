r"""Implementation of different Thole damping functions, and tools for plotting
and analyzing them

In Kirill's paper the damping functions are written in terms of two
factors: pow3_factor (lambda3) and pow5_factor (lambda5)

The functions that actually modify the dipole field tensor are the
derivatives of the potential, not the density or the potential itself.
"""
from numpy.typing import NDArray
import numpy as np


def exp_density(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho1 in Thole's paper
    """
    raise NotImplementedError()


def exp_density2(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho2 in Thole's paper
    """
    raise NotImplementedError()


def exp_density3(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho3 in Thole's paper
    """
    raise NotImplementedError()


def conical_density(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho4 in Thole's paper
    """
    raise NotImplementedError()


def conical_density2(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho5 in Thole's paper
    """
    raise NotImplementedError()


def conical_density3(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho6 in Thole's paper
    """
    raise NotImplementedError()


def conical_density4(
    reduced_dist: NDArray[np.floating], a_thole: float
) -> NDArray[np.floating]:
    r"""
    Function rho7 in Thole's paper
    """
    raise NotImplementedError()
