import numpy as np

from dipoles import calc_pair_dipole_field_matrix

d = 0.3
coords = np.array([[[0.0, 0, 0], [d, 0, 0]]])
alphas = np.array([[[4.50711, 4.50711]]]) * 0.14818471
# Homogeneous efield over all atoms
efield = np.array([[[1.0, 0, 0], [1.0, 0, 0]]])

# Calculation of dipole tensor by hand:
prefactor = -3 / (d**5)
array = np.array(
    [
        [
            [d**2 - 1 / 3 * d**2, 0, 0],
            [0, -1 / 3 * d**2, 0],
            [0, 0, -1 / 3 * d**2],
        ]
    ]
)
print(calc_pair_dipole_field_matrix(coords))
dipole_tensor_expect = prefactor * array
print(dipole_tensor_expect)


dx = 0.3
dy = 0.1
dz = 0.2
d = np.sqrt(dx**2 + dy**2 + dz**2)
coords = np.array([[[0.0, 0, 0], [dx, dy, dz]]])
alphas = np.array([[[4.50711, 4.50711]]]) * 0.14818471
# Homogeneous efield over all atoms
efield = np.array([[[1.0, 0, 0], [1.0, 0, 0]]])

# Calculation of dipole tensor by hand:
prefactor = -3 / (d**5)
array = np.array(
    [
        [
            [dx**2 - 1 / 3 * d**2, dx * dy, dx * dz],
            [dx * dy, dy**2 - 1 / 3 * d**2, dz * dy],
            [dx * dz, dz * dy, dz**2 - 1 / 3 * d**2],
        ]
    ]
)
print(calc_pair_dipole_field_matrix(coords))
dipole_tensor_expect = prefactor * array
print(dipole_tensor_expect)
