"""Main Tile Functions"""

from abc import ABC, abstractmethod
import inspect
from typing import List, Callable, Union
import matplotlib.pyplot as plt
import xarray as xr

import pystac
from .planetary import PCDownloader


class ImgTile(ABC):
    """ImgTile class"""

    thumb_shape = (512, 512)

    # ------ CONSTRUCTOR ------
    def __init__(self, stac_item: pystac.Item):
        self.stac_item = stac_item
        self._shape = None
        self._resolution = None
        self._thumb = None
        self.bands = {}

    # ------ Abstract Methods ------
    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Metadata tha is Satellite specific (e.g., bands names, shapes)"""

    @abstractmethod
    def get_mask(self) -> xr.DataArray:
        """Get the mask with invalid pixels for the specific sensor"""

    # ------ Properties ------
    @property
    def thumb(self):
        """Thumb"""
        if self._thumb is None:
            bands = self.convert_common_names(["Red", "Green", "Blue"])
            self._thumb = PCDownloader.get_assets(
                item=self.stac_item,
                assets=bands,
                shape=self.thumb_shape,
                scale=self.metadata["scale"],
            )

        return self._thumb

    @property
    def datestr(self):
        """Date"""
        return self.stac_item.datetime.strftime("%Y-%m-%d")

    @property
    def shape(self):
        """Shape"""
        if self._shape is not None:
            return self._shape
        else:
            return self.metadata["shape"]

    @shape.setter
    def shape(self, value):
        """Shape"""
        self._shape = value

    @property
    def resolution(self):
        """Calculate the resolution based on the current shape and the metadata shape"""
        resolution = (
            self.metadata["base_resolution"] * self.metadata["shape"][0] / self.shape[0]
        )

        return resolution

    @resolution.setter
    def resolution(self, resolution: int):
        """Set the output shape according to the given pixel resolution

        Args:
            resolution (int): Resolution of pixel in meters
        """
        size = int(self.metadata["shape"][0] * 10 / resolution)
        shape = (size, size)

        if shape != self.shape:
            self.bands.clear()
            self.shape = shape

    # ------ Private Met hods ------

    # ------ Public Methods ------
    def convert_common_names(self, common_names: List[str]):
        """Convert common names into band names"""

        # get the band names from the metadata
        bands_names = self.metadata["bands_names"]

        bands = [bands_names[name] for name in common_names]
        return bands

    def plot_thumb(self, ax: plt.Axes = None):
        """Plot thumb"""
        if not ax:
            _, ax = plt.subplots(figsize=(5, 5))

        self.thumb.plot.imshow(ax=ax, rgb="band", vmin=0, vmax=0.25)

    def get_band(self, band: str, scale: float = 1e-4, refresh: bool = False):
        """Get item"""

        if refresh or band not in self.bands:
            self.bands[band] = PCDownloader.get_asset(
                item=self.stac_item,
                asset=band,
                shape=self.shape,
                scale=scale,
            )

        return self.bands[band]

    def get_cube(self, bands: List[str]):
        """Get cube"""

        cube = xr.concat([self.get_band(b) for b in bands], dim="band")
        cube["band"] = list(bands)
        return cube

    def add_math_band(self, function: Callable):
        """Add math band"""
        func_args = inspect.signature(function)

        bands = [
            arg
            for arg in func_args.parameters
            if func_args.parameters[arg].default
            == inspect._empty  # pylint: disable=protected-access
        ]

        args = {b: self[b] for b in bands}

        name = function.__name__
        self.bands[name] = function(**args)

    # ------ Magic Methods ------

    def __getitem__(self, selector: Union[str, List[str]]):
        """Get item"""

        if isinstance(selector, str):
            return self.get_band(selector)
        else:
            return self.get_cube(selector)

    def __str__(self) -> str:
        return self.stac_item.properties["s2:mgrs_tile"]

    def __repr__(self) -> str:
        s = f"{self.stac_item.properties['s2:mgrs_tile']}({self.datestr})\n"

        if len(self.bands) > 0:
            s += f"Loaded bands: {list(self.bands.keys())}"
        return s

    def __setitem__(self, key, value):
        self.bands[key] = value
