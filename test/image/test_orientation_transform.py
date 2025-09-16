from logging import getLogger

import nibabel as nib
import numpy as np
import pytest
from numpy import typing as npt
from pytest import approx

from FastSurferCNN.data_loader.conform import (
    affine_for_target_orientation,
    does_vox2vox_rot_require_interpolation,
    ornt2affine,
)
from FastSurferCNN.utils.arg_types import OrientationType, StrictOrientationType

logger = getLogger(__name__)
SQRT1_2 = np.sqrt(0.5)
SQRT3_4 = np.sqrt(0.75)


@pytest.mark.parametrize(argnames=["shape"], argvalues=[[(1,) * 2]])
def test_ornt2affine_valueerror(shape: tuple[int]):
    """Test whether ornt2affine raises the correct ValueError"""
    with pytest.raises(ValueError, match="length of shape"):
        ornt2affine(np.transpose([np.arange(3), np.ones((3,))]), shape=shape)


@pytest.mark.parametrize(argnames="shape", argvalues=[None, (1, 1, 1)])
def test_ornt2affine_shape(shape):
    """Test whether ornt2affine returns the correct output shape."""
    actual = ornt2affine(np.transpose([np.arange(3), np.ones((3,))]), shape=shape).shape
    expected = (3 + int(shape is not None),) * 2
    assert actual == expected, "The shape of ornt2affine was incorrect!"


def test_ornt2affine_axcode(strict_orientation: StrictOrientationType):
    """Test whether ornt2affine returns the an affine of the correct axcode."""
    vox2vox = ornt2affine(nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS")), (0,) * 3)
    actual = "".join(nib.orientations.aff2axcodes(vox2vox, ("LR", "PA", "IS")))
    expected = strict_orientation
    assert actual == expected, "ornt2affine did not return a vox2vox of the correct orientation."


@pytest.mark.parametrize(argnames=["axcode", "translation"], argvalues=[["ALS", [1, 0, 0]], ["PIR", [0, 1, 1]]])
def test_ornt2affine_translation(axcode, translation, img_size: int):
    """Test whether the translation of ornt2affine is correct."""
    ornt = nib.orientations.axcodes2ornt(axcode)
    actual = ornt2affine(ornt, shape=(img_size,) * 3)[:3, 3]
    expected = np.asarray(translation) * (img_size - 1)
    assert actual == approx(expected), "Translation component of the vox2vox from ornt2affine did not match!"


@pytest.mark.parametrize(
    argnames=["axcode", "translation"],
    argvalues=[["ALS", [SQRT3_4, 0.5, 0]], ["PIR", [-0.5, SQRT3_4, 1]], ["PIL", [SQRT3_4 - 0.5, 0.5 + SQRT3_4, 1]]],
)
def test_ornt2affine_translation2(axcode: StrictOrientationType, translation: npt.ArrayLike, img_size: int):
    """Test whether the translation of ornt2affine is correct."""
    from scipy.spatial.transform import Rotation

    ornt = nib.orientations.axcodes2ornt(axcode)
    img_affine = np.pad(Rotation.from_euler("XYZ", [0, 0, 30], degrees=True).as_matrix(), ((0, 1), (0, 1)))
    img_affine[:, 3] = [0, 5, 0, 1]

    vox2vox = ornt2affine(ornt, shape=(img_size,) * 3)
    actual = (img_affine @ vox2vox)[:3, 3]
    expected = np.asarray(translation) * (img_size - 1) + img_affine[:3, 3]
    assert actual == approx(expected), "Translation component of the vox2vox from ornt2affine did not match!"


def test_ornt2affine_data(strict_orientation: OrientationType, img_size: int):
    """Test whether ornt2affine + apply_vox2vox equals scipy.ndimage.affine_transform."""
    from scipy.ndimage import affine_transform

    shape = (img_size,) * 3
    # not actually using the strict re-orientation
    ornt = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS"))
    vox2vox = ornt2affine(ornt, shape=shape)
    data = np.random.randn(*shape)
    expected = nib.orientations.apply_orientation(data, ornt)
    actual = affine_transform(data, vox2vox)
    assert actual == approx(expected), "affine_transform and apply_affine did not yield the same result!"


def test_affine_for_target_orientation(random_affine: npt.NDArray[float], img_size: int, orientation: OrientationType):
    """Test whether affine_for_target_orientation works as expected."""
    vox2vox = affine_for_target_orientation(random_affine, orientation, (img_size,) * 3)
    combined = random_affine @ vox2vox
    actual = "".join(nib.orientations.aff2axcodes(combined, ("lr", "pa", "is")))
    expected = orientation.lower().removeprefix("soft").lstrip("-_ ")
    assert actual == expected, f"affine_for_target_orientation did not yield a transformation to {orientation}!"


def test_affine_for_target_orientation2(
        random_affine: npt.NDArray[float],
        img_size: int,
        strict_orientation: StrictOrientationType,
):
    """Test whether affine_for_target_orientation works as expected."""
    vox2vox = affine_for_target_orientation(random_affine, "soft " + strict_orientation, (img_size,) * 3)
    is_reorder_flip = not does_vox2vox_rot_require_interpolation(vox2vox)
    assert is_reorder_flip, "affine_for_target_orientation did not yield a \"soft\" vox2vox transformation!"

#
# class TestAffineForTargetOrientation:
#     """Tests the function conform.orientation_to_ornts."""
#
#     TwoOrnts = tuple[npt.NDArray[int], npt.NDArray[int]]
#
#     @pytest.fixture(scope="class")
#     def output(self, random_affine: npt.NDArray[float], strict_orientation: StrictOrientationType) -> TwoOrnts:
#         return orientation_to_ornts(random_affine, strict_orientation)
#
#     def test_forward_and_back(self, output: TwoOrnts):
#         """Test whether the two outputs of orientation_to_ornts are inverse if each other."""
#         actual = nib.orientations.ornt_transform(*output)
#         expected = np.stack([np.arange(3), np.ones((3,))], axis=-1)
#         assert actual == approx(expected), "The two outputs of orientation_to_ornts are not the inverse of each other"
#
#     def test_axcode(
#             self,
#             output: TwoOrnts,
#             random_affine: npt.NDArray[float],
#             strict_orientation: StrictOrientationType,
#             img_size: int,
#     ):
#         """Test whether the axcodes of an affine transformed by reorient_affine are correct."""
#         from nibabel.orientations import ornt2axcodes, io_orientation
#         reoriented_affine = reorient_affine(random_affine, output[0], (img_size,) * 3)
#         actual = "".join(ornt2axcodes(io_orientation(reoriented_affine), ("LR", "PA", "IS")))
#         expected = strict_orientation
#         assert actual == expected, "Axcodes of affine after orientations_to_ornts + reorient_affine was not correct."
