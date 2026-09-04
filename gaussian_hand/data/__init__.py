"""HOT3D data preparation for the subject-bound Gaussian hand."""

from .geometry import crop_intrinsics, square_crop, visible_mano_ratio

__all__ = ["crop_intrinsics", "square_crop", "visible_mano_ratio"]
