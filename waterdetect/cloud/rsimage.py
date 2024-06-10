"""Docstring"""

from typing import Optional, List, Tuple, Union
from abc import ABC, abstractmethod

import pystac
from odc.stac import stac_load, extract_collection_metadata
import planetary_computer as pc

import xarray as xr


class RSImage(ABC):
    """_summary_"""

    stac_cfg = None

    def __init__(
        self, stac_items: List[pystac.Item], bbox: List[float], resolution: int = 10
    ):

        self.stac_items = stac_items
        self._metadata = extract_collection_metadata(stac_items[0])
        self.resolution = resolution
        self.bbox = bbox

        self.ds = stac_load(
            items=stac_items,
            bands=self._metadata.all_bands,
            resolution=resolution,
            patch_url=pc.sign,
            chunks={"time": 1, "x": 2048, "y": 2048},
            bbox=bbox,
            groupby="solar_day",
            stac_cfg=self.stac_cfg,
        )

    # ---------------------------------------------------------------------
    # Static methods
    # ---------------------------------------------------------------------
    @classmethod
    def get_tiling(cls) -> str:
        """Get the tiling system referenced in the Metadata"""
        if "tiling" in cls.metadata:
            tiling = cls.metadata["tiling"]
        else:
            tiling = "No tiling defined"

        return tiling

    @classmethod
    def extract_tile(cls, stac_item: pystac.Item) -> str:
        """Get the tile from a stac_item"""
        if "tile_property" not in cls.metadata:
            tile = ""
        else:
            tp = cls.metadata["tile_property"]

            if isinstance(tp, list):
                tile = "/".join([stac_item.properties[p] for p in tp])
            else:
                tile = stac_item.properties[tp]

        return tile

    # ---------------------------------------------------------------------
    # Abstract methods
    # ---------------------------------------------------------------------
    @abstractmethod
    def metadata(self) -> dict:
        """Metadata tha is Satellite specific (e.g., bands names, shapes)"""

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the image"""

        return self.ds[next(iter(self.ds.data_vars))].squeeze().shape

    def convert_common_names(self, common_names: List[str]):
        """Convert common names into band names"""

        # get the band names from the metadata
        bands_names = self.metadata["bands_names"]

        bands = [bands_names[name] for name in common_names]
        return bands

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------
    def get_band(self, band: str, scale: Optional[float] = None):
        """Get one band"""
        if band != self.metadata["qa_band"]:
            scale = scale or self.metadata["scale"]
            result = self.ds[band].astype("float16") * scale

        else:
            result = self.ds[band]

        return result.compute()

    def get_cube(self, bands: List[str]):
        """Get cube"""

        cube = xr.concat([self.get_band(b) for b in bands], dim="band")
        cube["band"] = list(bands)
        return cube

    def __getitem__(self, selector: Union[str, List[str]]):
        """Get item"""

        if isinstance(selector, str):
            return self.get_band(selector)
        else:
            return self.get_cube(selector)
