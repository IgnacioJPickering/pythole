import numpy as np

from pythole.dipoles import calc_pair_dipole_field_matrix

# Triatomic (water)
dx = 0.3
dy = 0.1
dz = 0.2
d = np.sqrt(dx**2 + dy**2 + dz**2)
coords = np.array([[[0.0, 0, 0], [dx, dy, dz]]])


coords = np.expand_dims(
    np.array(
        [
            [-0.001, 0.363, 0.000],
            [-0.825, -0.182, 0.000],
            [0.826, -0.181, 0.000],
        ]
    ),
    0,
)

water_dipole_field = calc_pair_dipole_field_matrix(coords)
oh1_diff = coords[0, 1] - coords[0, 0]
oh1x = oh1_diff[0]
oh1y = oh1_diff[1]
oh1z = oh1_diff[2]
oh1d = np.linalg.norm(oh1_diff)

oh2_diff = coords[0, 2] - coords[0, 0]
oh2x = oh2_diff[0]
oh2y = oh2_diff[1]
oh2z = oh2_diff[2]
oh2d = np.linalg.norm(oh2_diff)

hh_diff = coords[0, 2] - coords[0, 1]
hhx = hh_diff[0]
hhy = hh_diff[1]
hhz = hh_diff[2]
hhd = np.linalg.norm(hh_diff)
pair_inv_dist_expect = np.expand_dims(
    1 / np.array([[np.inf, oh1d, oh2d], [oh1d, np.inf, hhd], [oh2d, hhd, np.inf]]), 0
)

alphas = np.array([[[5.3, 4.50711, 4.50711]]]) * 0.14818471
zeros = np.zeros((1, 1, 1, 3, 3))

# Outer rr arrays
oh1_outer_rr = np.array(
    [
        [
            [oh1x**2, oh1x * oh1y, oh1x * oh1z],
            [oh1x * oh1y, oh1y**2, oh1z * oh1y],
            [oh1x * oh1z, oh1z * oh1y, oh1z**2],
        ]
    ]
).reshape(1, 1, 1, 3, 3)
oh2_outer_rr = np.array(
    [
        [
            [oh2x**2, oh2x * oh2y, oh2x * oh2z],
            [oh2x * oh2y, oh2y**2, oh2z * oh2y],
            [oh2x * oh2z, oh2z * oh2y, oh2z**2],
        ]
    ]
).reshape(1, 1, 1, 3, 3)
hh_outer_rr = np.array(
    [
        [
            [hhx**2, hhx * hhy, hhx * hhz],
            [hhx * hhy, hhy**2, hhz * hhy],
            [hhx * hhz, hhz * hhy, hhz**2],
        ]
    ]
).reshape(1, 1, 1, 3, 3)
# Seems like all the elements of the outer_rr matrix match

# Dipole tensor field arrays
oh1_array = (
    np.array(
        [
            [
                [oh1x**2 - 1 / 3 * oh1d**2, oh1x * oh1y, oh1x * oh1z],
                [oh1x * oh1y, oh1y**2 - 1 / 3 * oh1d**2, oh1z * oh1y],
                [oh1x * oh1z, oh1z * oh1y, oh1z**2 - 1 / 3 * oh1d**2],
            ]
        ]
    ).reshape(1, 1, 1, 3, 3)
    * -3
    / (oh1d**5)
)
oh2_array = (
    np.array(
        [
            [
                [oh2x**2 - 1 / 3 * oh2d**2, oh2x * oh2y, oh2x * oh2z],
                [oh2x * oh2y, oh2y**2 - 1 / 3 * oh2d**2, oh2z * oh2y],
                [oh2x * oh2z, oh2z * oh2y, oh2z**2 - 1 / 3 * oh2d**2],
            ]
        ]
    ).reshape(1, 1, 1, 3, 3)
    * -3
    / (oh2d**5)
)
hh_array = (
    np.array(
        [
            [
                [hhx**2 - 1 / 3 * hhd**2, hhx * hhy, hhx * hhz],
                [hhx * hhy, hhy**2 - 1 / 3 * hhd**2, hhz * hhy],
                [hhx * hhz, hhz * hhy, hhz**2 - 1 / 3 * hhd**2],
            ]
        ]
    ).reshape(1, 1, 1, 3, 3)
    * -3
    / (hhd**5)
)


hyperrow_1 = np.concatenate((zeros, oh1_array, oh2_array), axis=1)

hyperrow_2 = np.concatenate((oh1_array, zeros, hh_array), axis=1)

hyperrow_3 = np.concatenate((oh2_array, hh_array, zeros), axis=1)

water_dipole_field_expect = np.concatenate((hyperrow_1, hyperrow_2, hyperrow_3), axis=2)
diff = water_dipole_field - water_dipole_field_expect
# Seems that these two are exactly equal
diff[np.abs(diff) < 1e-10] = 0
assert (diff == 0).all()
