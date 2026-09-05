import numpy as np

from gaussian_hand.data.geometry import crop_intrinsics, square_crop, visible_mano_ratio


def test_visible_ratio_uses_complete_mano_as_denominator():
    mano = np.zeros((8, 8), np.uint8)
    mano[2:6, 2:6] = 1
    sam = np.ones((8, 8), np.uint8)
    sam[2:4, 2:6] = 0
    assert visible_mano_ratio(sam, mano) == 0.5


def test_square_crop_stays_inside_image():
    crop = square_crop(np.array([[1.0, 5.0], [21.0, 15.0]]), 30, 20, 1.2)
    assert crop is not None
    x0, y0, x1, y1 = crop
    assert x0 >= 0 and y0 >= 0 and x1 <= 30 and y1 <= 20
    assert abs((x1 - x0) - (y1 - y0)) <= 1


def test_crop_intrinsics_preserves_projected_pixel_centres():
    k = np.array([[100.0, 0.0, 40.0], [0.0, 120.0, 50.0], [0.0, 0.0, 1.0]])
    result = crop_intrinsics(k, (10, 20, 110, 120), 200)
    assert np.allclose(result, [[200, 0, 60.5], [0, 240, 60.5], [0, 0, 1]])


def test_silhouette_is_union_of_overlapping_triangles():
    from gaussian_hand.data.geometry import rasterize_silhouette
    uv = np.array([[2., 2.], [18., 2.], [2., 18.]])
    z = np.ones(3)
    # Front and back mesh faces may have the same projected footprint.
    single = rasterize_silhouette(uv, z, np.array([[0, 1, 2]]), 24, 24, 1)
    overlap = rasterize_silhouette(uv, z, np.array([[0, 1, 2], [0, 2, 1]]), 24, 24, 1)
    np.testing.assert_array_equal(overlap, single)
    assert overlap[5, 5] == 1
