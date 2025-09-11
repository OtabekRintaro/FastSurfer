from logging import getLogger

import nibabel as nib
import numpy as np
from numpy import typing as npt
import pytest
from pytest import approx

from FastSurferCNN.data_loader.conform import orientation_to_ornts, reorient_affine, inv_ornt, concat_ornts, \
    ornt_to_ornt, to_target_orientation
from FastSurferCNN.utils.arg_types import StrictOrientationType, OrientationType

logger = getLogger(__name__)
SQRT1_2 = np.sqrt(0.5)
SQRT3_4 = np.sqrt(0.75)


def test_reorient_affine_typeerror():
     """Test whether reorient_affine raises TypeError exceptions if vox_size is missing."""
     with pytest.raises(TypeError, match="shape"):
        reorient_affine(np.eye(4, dtype=float), np.stack([np.arange(3), -np.ones((3,))], axis=-1))


def test_inv_ornt(strict_orientation: StrictOrientationType):
    """Test whether inv_ornt works correctly."""
    from nibabel.orientations import axcodes2ornt, aff2axcodes
    labels = ("LR", "PA", "IS")
    ornt = axcodes2ornt(strict_orientation, labels)
    inverted_affine = np.linalg.inv(reorient_affine(np.eye(4), ornt, (1,) * 3))
    expected = np.asarray(axcodes2ornt(aff2axcodes(inverted_affine, labels), labels), dtype=int)
    actual = inv_ornt(ornt)
    assert actual == approx(expected), "Inversion of ornts did not match!"


def test_ornt_to_ornt(random_affine: npt.NDArray[float], strict_orientation: StrictOrientationType, img_size: int):
    """Test whether ornt_to_ornt works correctly."""
    from nibabel.orientations import io_orientation, axcodes2ornt
    labels = ("LR", "PA", "IS")
    expected = axcodes2ornt(strict_orientation, labels)
    expected_affine = reorient_affine(np.eye(4), expected, (img_size,) * 3)
    target_ornt = io_orientation(random_affine @ expected_affine)
    actual = ornt_to_ornt(io_orientation(random_affine), target_ornt)
    assert actual == approx(expected), "The ornt_to_ornt array did not agree with calculation via affines!"


@pytest.mark.parametrize(argnames=["axcode", "translation"], argvalues=[["ALS", [1, 0, 0]], ["PIR", [0, 1, 1]]])
def test_reorient_affine_translation(axcode, translation, img_size: int):
    """Test whether the translation of reorient_affine is correct."""
    ornt = nib.orientations.axcodes2ornt(axcode)
    aff = reorient_affine(np.eye(4, dtype=float), ornt, shape=(img_size,) * 3)
    actual = aff[:3, 3]
    expected = np.asarray(translation) * (img_size - 1)
    assert actual == approx(expected), "Affine did not match"


@pytest.mark.parametrize(
    argnames=["axcode", "translation"],
    argvalues=[["ALS", [SQRT3_4, 0.5, 0]], ["PIR", [-0.5, SQRT3_4, 1]], ["PIL", [SQRT3_4 - 0.5, 0.5 + SQRT3_4, 1]]],
)
def test_reorient_affine_translation2(axcode: StrictOrientationType, translation: npt.ArrayLike, img_size: int):
    """Test whether the translation of reorient_affine is correct."""
    from scipy.spatial.transform import Rotation

    ornt = nib.orientations.axcodes2ornt(axcode)
    aff = np.pad(Rotation.from_euler("XYZ", [0, 0, 30], degrees=True).as_matrix(), ((0, 1), (0, 1)))
    aff[:, 3] = [0, 5, 0, 1]

    affine = reorient_affine(aff, ornt, shape=(img_size,) * 3)
    actual = affine[:3, 3]
    expected = np.asarray(translation) * (img_size - 1) + aff[:3, 3]
    assert actual == approx(expected), "Affine did not match"


def test_reorient_affine_data(random_affine: npt.NDArray, strict_orientation: OrientationType, img_size: int):
    """Test whether reorient_affine + nibabel.orientations.apply_orientation equals scipy.ndimage.affine_transform."""
    from scipy.ndimage import affine_transform

    shape = (img_size,) * 3
    centered_affine = random_affine.copy()
    centered_affine[:3, 3] = 0
    # not actually using the strict re-orientation
    ornt = nib.orientations.axcodes2ornt(strict_orientation, ("LR", "PA", "IS"))
    transformed_affine = reorient_affine(centered_affine, ornt, shape=shape)
    data = np.random.randn(*shape)
    expected = nib.orientations.apply_orientation(data, ornt)
    actual = affine_transform(data, centered_affine @ np.linalg.inv(transformed_affine))
    assert actual == approx(expected), "affine_transform and apply_affine did not yield the same result!"

class TestOrientationToOrnts:
    """Tests the function conform.orientation_to_ornts."""

    TwoOrnts = tuple[npt.NDArray[int], npt.NDArray[int]]

    @pytest.fixture(scope="class")
    def output(self, random_affine: npt.NDArray[float], strict_orientation: StrictOrientationType) -> TwoOrnts:
        return orientation_to_ornts(random_affine, strict_orientation)

    def test_forward_and_back(self, output: TwoOrnts):
        """Test whether the two outputs of orientation_to_ornts are inverse if each other."""
        actual = nib.orientations.ornt_transform(*output)
        expected = np.stack([np.arange(3), np.ones((3,))], axis=-1)
        assert actual == approx(expected), "The two oututs of orientation_to_ornts are not the inverse of each other."

    def test_axcode(
            self,
            output: TwoOrnts,
            random_affine: npt.NDArray[float],
            strict_orientation: StrictOrientationType,
            img_size: int,
    ):
        """Test whether the axcodes of an affine transformed by reorient_affine are correct."""
        from nibabel.orientations import ornt2axcodes, io_orientation
        reoriented_affine = reorient_affine(random_affine, output[0], (img_size,) * 3)
        actual = "".join(ornt2axcodes(io_orientation(reoriented_affine), ("LR", "PA", "IS")))
        expected = strict_orientation
        assert actual == expected, "Axcodes of affine after orientations_to_ornts + reorient_affine was not correct."
