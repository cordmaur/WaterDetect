"""This module defines the template of a Sentinel-2 Image"""

from typing import List
import xarray as xr
from .rsimage import RSImage


class S2Image(RSImage):
    """_summary_"""

    collection = "sentinel-2-l2a"

    # data type configuration for the layers
    stac_cfg = {
        "sentinel-2-l2a": {
            "assets": {
                "*": {"data_type": "uint16", "nodata": 0},
                "SCL": {"data_type": "uint8", "nodata": 0},
                # "visual": {"data_type": "uint8", "nodata": 0},
            }
        }
    }

    metadata = {
        "bands_names": {
            "Blue": "B02",
            "Green": "B03",
            "Red": "B04",
            "Mir": "B11",
            "Mir2": "B12",
            "Nir": "B08",
            "Nir2": "B8A",
            "RedEdg1": "B05",
            "RedEdg2": "B06",
            "RedEdg3": "B07",
        },
        "scale": 1e-4,
        "base_resolution": 10,
        "shape": (10980, 10980),
        "qa_band": "SCL",
        "tiling": "MGRS",
        "tile_property": "s2:mgrs_tile",
    }

    Sen2CorMaskList = {
        "NO_DATA": 0,
        "SATURATED_OR_DEFECTIVE": 1,
        "DARK_AREA_PIXELS": 2,
        "CLOUD_SHADOWS": 3,
        "VEGETATION": 4,
        "NOT_VEGETATED": 5,
        "WATER": 6,
        "UNCLASSIFIED": 7,
        "CLOUD_MEDIUM_PROBABILITY": 8,
        "CLOUD_HIGH_PROBABILITY": 9,
        "THIN_CIRRUS": 10,
        "SNOW": 11,
    }

    # Overviews: 5490x5490, 2745x2745, 1373x1373, 687x687, 344x344
    def get_mask(self, mask_list: List[str]) -> xr.DataArray:
        """
        Get the mask for the Sentinel-2 sensor, considering the SCL layer

        Args:
            mask_list (List[str]): List of masks to consider

        Returns:
            _type_: xr.DataArray
        """
        scl = self.get_band(self.metadata["qa_band"], scale=1)

        mask = xr.zeros_like(scl)
        for mask_name in mask_list:
            mask |= scl == self.Sen2CorMaskList[mask_name.upper()]

        return mask.astype("bool").squeeze()
