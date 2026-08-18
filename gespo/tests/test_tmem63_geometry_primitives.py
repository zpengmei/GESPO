import numpy as np

from gespo.utils.geometry import (
    angle_between_vectors,
    angle_to_axis_from_xyz,
    axis_from_xyz_pca,
    calc_angle,
    centroid_from_xyz,
    mean_pairwise_distance_from_xyz,
    pairwise_distance_from_xyz,
)


def test_centroid_from_xyz_averages_selected_atoms():
    xyz = np.zeros((2, 4, 3), dtype=np.float64)
    xyz[0, [0, 1]] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    xyz[1, [0, 1]] = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]

    out = centroid_from_xyz(xyz, np.array([0, 1]))

    np.testing.assert_allclose(out, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], atol=1e-5)


def test_centroid_from_xyz_empty_group_returns_zeros():
    xyz = np.random.default_rng(0).normal(size=(3, 5, 3))
    out = centroid_from_xyz(xyz, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(out, np.zeros((3, 3), dtype=np.float32))


def test_pairwise_distance_from_xyz_matches_known_distance():
    xyz = np.zeros((1, 2, 3), dtype=np.float64)
    xyz[0, 1] = [3.0, 4.0, 0.0]
    out = pairwise_distance_from_xyz(xyz, 0, 1)
    np.testing.assert_allclose(out, [5.0], atol=1e-5)


def test_mean_pairwise_distance_within_single_group():
    # Three points on a line at 0, 1, 3 -> pairwise dists 1, 3, 2 -> mean 2.0
    xyz = np.zeros((1, 3, 3), dtype=np.float64)
    xyz[0, :, 0] = [0.0, 1.0, 3.0]
    out = mean_pairwise_distance_from_xyz(xyz, np.array([0, 1, 2]))
    np.testing.assert_allclose(out, [2.0], atol=1e-5)


def test_mean_pairwise_distance_single_group_too_small_is_zero():
    xyz = np.zeros((1, 3, 3), dtype=np.float64)
    out = mean_pairwise_distance_from_xyz(xyz, np.array([0]))
    np.testing.assert_allclose(out, [0.0])


def test_mean_pairwise_distance_between_two_groups():
    # group a = {x=0}, group b = {x=2, x=4} -> distances 2, 4 -> mean 3.0
    xyz = np.zeros((1, 3, 3), dtype=np.float64)
    xyz[0, :, 0] = [0.0, 2.0, 4.0]
    out = mean_pairwise_distance_from_xyz(xyz, np.array([0]), np.array([1, 2]))
    np.testing.assert_allclose(out, [3.0], atol=1e-5)


def test_mean_pairwise_distance_empty_group_returns_zero():
    xyz = np.zeros((2, 3, 3), dtype=np.float64)
    out = mean_pairwise_distance_from_xyz(xyz, np.array([], dtype=np.int64), np.array([0]))
    np.testing.assert_array_equal(out, np.zeros((2,), dtype=np.float32))


def test_angle_between_vectors_right_angle():
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0]])
    out = angle_between_vectors(v1, v2)
    np.testing.assert_allclose(out, [np.pi / 2], atol=1e-5)


def test_angle_between_vectors_parallel_and_antiparallel():
    v1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v2 = np.array([[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    out = angle_between_vectors(v1, v2)
    np.testing.assert_allclose(out, [0.0, np.pi], atol=1e-5)


def test_angle_between_vectors_zero_vector_is_defined_as_zero():
    v1 = np.array([[0.0, 0.0, 0.0]])
    v2 = np.array([[1.0, 0.0, 0.0]])
    out = angle_between_vectors(v1, v2)
    np.testing.assert_allclose(out, [0.0])


def test_calc_angle_right_triangle_vertex_at_b():
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    out = calc_angle(a, b, c)
    np.testing.assert_allclose(out, [np.pi / 2], atol=1e-5)


def test_axis_from_xyz_pca_recovers_line_direction_and_sign():
    # Colinear points along +x, in increasing order -> axis should point in +x.
    xyz = np.zeros((1, 5, 3), dtype=np.float64)
    xyz[0, :, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]
    axis = axis_from_xyz_pca(xyz, np.arange(5))
    np.testing.assert_allclose(axis[0], [1.0, 0.0, 0.0], atol=1e-4)

    # Reversing the index order should flip the recovered sign too, since the
    # sign is anchored to first-half -> second-half order of `atom_idx`.
    axis_rev = axis_from_xyz_pca(xyz, np.arange(4, -1, -1))
    np.testing.assert_allclose(axis_rev[0], [-1.0, 0.0, 0.0], atol=1e-4)


def test_axis_from_xyz_pca_too_few_atoms_returns_zero():
    xyz = np.zeros((2, 5, 3), dtype=np.float64)
    axis = axis_from_xyz_pca(xyz, np.array([0]))
    np.testing.assert_array_equal(axis, np.zeros((2, 3), dtype=np.float32))


def test_angle_to_axis_from_xyz_membrane_normal():
    xyz = np.zeros((2, 5, 3), dtype=np.float64)
    # Frame 0: helix segment runs along z -> ~0 angle to the z-axis.
    xyz[0, :, 2] = [0.0, 1.0, 2.0, 3.0, 4.0]
    # Frame 1: helix segment runs along x -> ~90 degrees to the z-axis.
    xyz[1, :, 0] = [0.0, 1.0, 2.0, 3.0, 4.0]

    out = angle_to_axis_from_xyz(xyz, np.arange(5), axis=(0.0, 0.0, 1.0))

    np.testing.assert_allclose(out, [0.0, np.pi / 2], atol=1e-4)
