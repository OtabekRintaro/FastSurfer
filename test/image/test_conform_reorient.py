from logging import getLogger
from typing import TypedDict

import nibabel as nib
import numpy as np
import pytest
from pytest import approx

from FastSurferCNN.data_loader.conform import OrientationType, conform, prepare_mgh_header
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, nibabelHeader
from FastSurferCNN.utils.arg_types import StrictOrientationType

logger = getLogger(__name__)

class MultiCoordImages(TypedDict):
    X: nib.Nifti1Image
    Y: nib.Nifti1Image
    Z: nib.Nifti1Image

conform_reorient = {"rescale": None, "dtype": np.float32}


def circle_data(img_size: int) -> Image3d:
    """Generates a 3D image with a centered sphere of radius img_size/2."""
    data = np.mgrid[0:img_size, 0:img_size, 0:img_size].astype(np.float32) - (img_size - 1) / 2.0
    return (np.sum(data * data, axis=0) < img_size * img_size / 4.0).astype(np.float32)


@pytest.fixture(scope="session")
def circle_image(random_affine: AffineMatrix4x4, img_size: int) -> nib.Nifti1Image:
    return nib.Nifti1Image(circle_data(img_size), random_affine)


@pytest.fixture(scope="session")
def random_image(random_affine: AffineMatrix4x4, img_size: int) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.random.randn(img_size, img_size, img_size), random_affine)


@pytest.fixture(scope="session")
def empty_image(random_affine: AffineMatrix4x4, img_size: int) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.empty((img_size,) * 3), random_affine)


@pytest.fixture(scope="session")
def worldcoord_images(random_affine: AffineMatrix4x4, img_size: int) -> MultiCoordImages:
    data = worldcoords_data(random_affine, img_size)
    return MultiCoordImages(**{c: nib.Nifti1Image(data[..., i], random_affine) for i, c in enumerate("XYZ")})


def worldcoords_data(affine: AffineMatrix4x4, img_size: int) -> np.ndarray:
    xi = np.moveaxis(np.mgrid[0:img_size, 0:img_size, 0:img_size], 0, -1)
    return nib.affines.apply_affine(affine, xi.reshape((-1, 3)).astype(float)).reshape(xi.shape).astype(np.float32)


def affine2orientation(affine: AffineMatrix4x4) -> OrientationType:
    """Generates the orientation type string from an affine matrix."""
    from nibabel.orientations import aff2axcodes

    orientation: StrictOrientationType = "".join(aff2axcodes(affine, ("LR", "PA", "IS")))
    if np.allclose(np.sum([np.isclose(np.abs(affine), i) for i in (0, 1.)], axis=0), 1):
        return orientation
    else:
        return "soft " + orientation


class HeaderTests:
    def test_affine_orientation(self, affine: AffineMatrix4x4, orientation: OrientationType):
        """Tests whether a conformed image actually has the correct orientation."""
        actual = affine2orientation(affine)
        expected = orientation
        assert actual == expected, "The expected orientation did not match the actual orientation."

    def test_affine_vox_size(self, affine: AffineMatrix4x4, vox_size: float):
        """Tests whether a conformed image actually has the correct voxel size."""
        actual = np.linalg.norm(affine[:3, :3], axis=0)
        expected = vox_size
        assert actual == approx(expected), "The actual voxel sizes in the affine did not match the expected."

    def test_vox_size(self, header: nibabelHeader, vox_size: float):
        """Tests whether a conformed image actually has the correct voxel size."""
        actual = header.get_zooms()
        expected = np.full_like(actual, vox_size)
        assert actual == approx(expected), "The actual voxel sizes in the affine did not match the expected."


class TestPrepareHeader(HeaderTests):

    @pytest.fixture(scope="class")
    def header(self, empty_image: nib.Nifti1Image, orientation: OrientationType, img_size: int, vox_size: float) \
            -> nib.freesurfer.mghformat.MGHHeader:
        return prepare_mgh_header(empty_image, vox_size, img_size, orientation)

    @pytest.fixture(scope="class")
    def affine(self, header: nib.freesurfer.mghformat.MGHHeader) -> AffineMatrix4x4:
        return header.get_affine()


class TestConformAffine(HeaderTests):

    @pytest.fixture(scope="class")
    def image(self, empty_image: nib.Nifti1Image, orientation: OrientationType, vox_size: float) -> nib.Nifti1Image:
        return conform(empty_image, orientation=orientation, vox_size=vox_size, **conform_reorient)

    @pytest.fixture(scope="class")
    def affine(self, image: nib.Nifti1Image) -> AffineMatrix4x4:
        return image.affine

    @pytest.fixture(scope="class")
    def header(self, image: nib.Nifti1Image) -> nib.Nifti1Header:
        return image.header


class TestThereAndBack:

    @pytest.fixture(scope="class")
    def image(self, circle_image: nib.Nifti1Image, soft_orientation: OrientationType) -> nib.MGHImage:
        from nibabel import aff2axcodes

        there = conform(circle_image, orientation=soft_orientation, **conform_reorient)
        in_orientation: OrientationType = "soft " + "".join(aff2axcodes(circle_image.affine, ("LR", "PA", "IS")))
        return conform(there, orientation=in_orientation, **conform_reorient)

    def test_affine(self, image: nib.MGHImage, circle_image: nib.Nifti1Image) -> None:
        """
        Tests whether the affines of the original and the re-oriented images are the same.
        """
        expected_affine = circle_image.affine
        assert image.affine == approx(expected_affine), "The affines of original and re-reoriented images differ!"

    def test_image(self, image: nib.MGHImage, circle_image: nib.Nifti1Image):
        """
        Tests whether the content of the original and the re-oriented images are the same in the "center circle".
        """

        # this has to be filtered by the region of the image that is the same
        inner_circle = circle_data(circle_image.shape[0] - 2)
        outer_circle = circle_data(circle_image.shape[0] + 2)[1:-1, 1:-1, 1:-1]
        mask = np.pad(inner_circle, 1, constant_values=0) + (1 - outer_circle)
        expected = np.where(mask, circle_image.dataobj, np.nan)
        back_data = np.asarray(image.dataobj)
        assert back_data == approx(expected, abs=1e-3), "The data differs from the re-oriented image!"


def test_reorient_worldcoords(worldcoord_images: MultiCoordImages, soft_orientation: OrientationType, vox_size: float):
    """
    This test checks, whether the world coordinates are consistent.

    In effect this means generating world coordinates for an original image (xyz) and

    Parameters
    ----------
    worldcoord_images

    Returns
    -------

    """
    from pytest import approx

    from FastSurferCNN.data_loader.conform import conform

    ximg, yimg, zimg = {k: conform(img, **conform_reorient) for k, img in worldcoord_images.items()}
    xyz = np.stack([ximg, yimg, zimg], axis=-1)
    logger.info("Checking affines of world images:")

    worldcoords = worldcoords_data(ximg.affine, ximg.shape[0])
    center_mask = np.pad(circle_data(ximg.shape[0] - 2), 1, constant_values=0)
    expected = np.where(center_mask, worldcoords, np.NaN)

    assert xyz == approx(expected, abs=0.1), "The world images "
