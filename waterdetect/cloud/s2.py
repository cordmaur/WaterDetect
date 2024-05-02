"""Waterdetect-cloud module"""

from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Callable
import inspect

import pandas as pd

import matplotlib.pyplot as plt

import pystac

import xarray as xr
import rioxarray as xrio

from .planetary import PCDownloader


class S2Tile:
    """S2Tile class"""

    thumb_shape = (512, 512)
    shape = (10980, 10980)

    def __init__(self, s2item: pystac.Item):
        self.item = s2item

        self._thumb = None
        self.bands = {}

    # ------ Properties ------
    @property
    def thumb(self):
        """Thumb"""
        if self._thumb is None:
            self._thumb = PCDownloader.get_assets(
                item=self.item,
                assets=["B04", "B03", "B02"],
                shape=S2Tile.thumb_shape,
                scale=1e-4,
            )

        return self._thumb

    @property
    def datestr(self):
        """Date"""
        return self.item.datetime.strftime("%Y-%m-%d")

    # ------ Private Methods ------

    # ------ Public Methods ------
    def plot_thumb(self, ax: plt.Axes = None):
        """Plot thumb"""
        if not ax:
            _, ax = plt.subplots(figsize=(5, 5))

        self.thumb.plot.imshow(ax=ax, rgb="band", vmin=0, vmax=0.25)

    def get_band(self, band: str):
        """Get item"""

        if band not in self.bands:
            self.bands[band] = PCDownloader.get_asset(
                item=self.item,
                asset=band,
                shape=S2Tile.shape,
                scale=1e-4,
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
        return self.item.properties["s2:mgrs_tile"]

    def __repr__(self) -> str:
        s = f"{self.item.properties['s2:mgrs_tile']}({self.datestr})\n"

        if len(self.bands) > 0:
            s += f"Loaded bands: {list(self.bands.keys())}"
        return s

    def __setitem__(self, key, value):
        self.bands[key] = value


class S2TileDownloader:
    """S2TileDownloader class"""

    collection = "sentinel-2-l2a"

    def __init__(self):
        self.downloader = PCDownloader()
        self.items = None

    # ------ Private Methods ------
    @staticmethod
    def _plot_thumb(idx: int, tile: S2Tile, selected: bool, ax: plt.Axes):
        """Plot thumb"""
        tile.plot_thumb(ax=ax)
        dt = tile.item.datetime.strftime("%Y-%m-%d")
        cloud_cover = round(tile.item.properties["eo:cloud_cover"], 2)

        title = f"Id: {idx} {'- selected' if selected else ''}"
        title += f"\n{dt} - {cloud_cover}%"
        ax.set_title(title)

    @staticmethod
    def _adjust_borders(axs: List[plt.Axes], select_list: List[bool]):
        # once the images plotted, let's delete the axis
        for idx, ax in enumerate(axs):
            if not ax.has_data():
                ax.axis("off")
            else:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                selected = select_list[idx]
                color = "green" if selected else "red"

                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(1.5)

    # ------ Public Methods ------
    @property
    def selected_tiles(self):
        """Selected tiles"""
        return self.items[self.items["Selected"]]["S2Tile"].to_list()

    def search_tile(self, tile: str, time_range: str, cloud_cover: int = 100):
        """Search tile"""
        query = {"eo:cloud_cover": {"lt": cloud_cover}, "s2:mgrs_tile": {"eq": tile}}

        items = self.downloader.search(
            query=query,
            datetime=time_range,
            collections=S2TileDownloader.collection,
        )

        print(
            "You can access them through `.items` and see their thumbnails (.plot_thumbs())"
        )

        # convert the items to a simple Pandas DataFrame
        d = {
            i: {
                "Datetime": i.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "CloudCover": round(i.properties["eo:cloud_cover"], 2),
                # "Tile": tile,
                "S2Tile": S2Tile(i),
            }
            for i in items
        }
        df = pd.DataFrame(d).T
        df.index.name = "Item"
        df = df.reset_index(drop=False)
        df["Selected"] = True

        self.items = df  # .reset_index(drop=False)

    def plot_thumbs(self, cols: int = 4, cell_size: int = 4, debug: bool = False):
        """
        Plot thumbs

        Args:
            cols (int, optional): _description_. Defaults to 4.
            cell_size (int, optional): _description_. Defaults to 5.
        """

        n = len(self.items)
        rows = n // cols + (1 if n % cols > 0 else 0)

        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * cell_size, rows * cell_size)
        )

        axs = axs.flatten()

        futures = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            for idx, (_, row) in enumerate(self.items.iterrows()):
                ax = axs[idx]

                # Create a partial function to plot the thumbnail in ax
                f = partial(
                    S2TileDownloader._plot_thumb,
                    idx=idx,
                    tile=row["S2Tile"],
                    selected=row["Selected"],
                    ax=ax,
                )

                if debug:
                    f()
                else:
                    futures.append(executor.submit(f))

        fig.suptitle(f"S2 tiles ({n})")

        # adjust the borders
        S2TileDownloader._adjust_borders(axs, self.items["Selected"].to_list())

        self.futures = futures

    def select_scenes(self, scenes: List[int]):
        """Select scenes"""
        self.items["Selected"] = False
        self.items.loc[scenes, "Selected"] = True

    def __getitem__(self, selector: Union[int, slice, List[int]]):
        """Get item"""
        if isinstance(selector, int):
            return self.items.loc[selector, "S2Tile"]
        else:
            return self.items.loc[selector, "S2Tile"].to_list()
