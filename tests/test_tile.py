"""Test the ImgTile class"""

import pytest
import pystac
import xarray as xr

from waterdetect.cloud.tile import ImgTile


class MockItemClass:
    """
    A mock class that mimics the behavior of the pystac.Item class.
    """


class MockTileClass(ImgTile):
    """
    A mock subclass of the ImgTile class that provides predefined metadata.
    """

    metadata = {
        "base_resolution": 10,
        "shape": (10980, 10980),
        "bands_names": {
            "Blue": "B02",
            "Green": "B03",
            "Red": "B04",
        },
        "scale": 1e-4,
    }


@pytest.fixture(name="stac_item")
def fixture_stac_item():
    """
    Create a mock pystac.Item object for testing purposes.

    Returns:
        A mock pystac.Item object with a predefined datetime value.
    """
    # Create a mock pystac.Item object
    url = """
    https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20200828T134219_R124_T21JWK_20201027T091350
    """
    item = pystac.Item.from_file(href=url)

    return item


@pytest.fixture(name="img_tile")
def fixture_img_tile(stac_item):
    """
    Create a mock ImgTile object for testing purposes.

    Returns:
        A mock ImgTile object with a predefined datetime value.
    """
    # Create a mock ImgTile object
    img_tile = MockTileClass(stac_item=stac_item)

    return img_tile


class TestImgTile:
    """
    Test suite for the ImgTile class.
    """

    def test_properties(self, img_tile):
        """
        Test the behavior of the ImgTile class when dealing with properties
        shape, resolucion, date, etc.

        Args:
            mock_stac_item (pystac.Item): A mock STAC item object.
        """

        # Test initial shape and resolution
        assert img_tile.shape == (10980, 10980)
        assert img_tile.resolution == 10.0

        # Set resolution and check shape
        img_tile.resolution = 20
        assert img_tile.shape == (5490, 5490)

        # Set shape and test resolution
        img_tile.shape = (10980, 10980)
        assert img_tile.resolution == 10

        # Test datestr property
        assert img_tile.datestr == "2020-08-28"

        # Test thumb
        thumb = img_tile.thumb
        assert isinstance(thumb, xr.DataArray)

    def test_get_band(self, img_tile):
        """
        Test the behavior of the get_band method.
        """
        img_tile.shape = (344, 344)
        band = img_tile.get_band(band="SCL", scale=1)

        assert "SCL" in img_tile.bands
        assert band.shape == (344, 344)
