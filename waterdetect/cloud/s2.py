"""Waterdetect-cloud module"""

from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import pandas as pd

import matplotlib.pyplot as plt

import xarray as xr
import rioxarray as xrio  # pylint: disable=W0611

from .tile import ImgTile
from .planetary import PCDownloader


class S2Tile(ImgTile):
    """S2Tile class"""

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
        "product_type": "S2_PLANETARY",
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

        return mask.astype('bool')


class S2TileDownloader:
    """S2TileDownloader class"""

    collection = "sentinel-2-l2a"

    def __init__(self):
        self.downloader = PCDownloader()
        self.items = self.items_df = None
        self.futures = []

    # ------ Private Methods ------
    @staticmethod
    def _plot_thumb(idx: int, tile: S2Tile, selected: bool, ax: plt.Axes):
        """Plot thumb"""
        tile.plot_thumb(ax=ax)
        dt = tile.stac_item.datetime.strftime("%Y-%m-%d")
        cloud_cover = round(tile.stac_item.properties["eo:cloud_cover"], 2)

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

        print("You can access them through `.items` or `.items_df`")
        print("You can also select them and see their thumbnails (.plot_thumbs())")

        # convert the items to a simple Pandas DataFrame
        d = {
            i.id: {
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

        self.items_df = df
        self.items = list(items)

    def plot_thumbs(self, cols: int = 4, cell_size: int = 4, debug: bool = False):
        """
        Plot thumbs

        Args:
            cols (int, optional): Number of columns in the thumbs grid. Defaults to 4.
            cell_size (int, optional): Size of each thumbnail. Defaults to 5.
            debug (bool): If True, make calls syncronously. Defaults to False.
        """

        n = len(self.items)
        rows = n // cols + (1 if n % cols > 0 else 0)

        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * cell_size, rows * cell_size)
        )

        axs = axs.flatten()

        futures = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            for idx, (_, row) in enumerate(self.items_df.iterrows()):
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
        S2TileDownloader._adjust_borders(axs, self.items_df["Selected"].to_list())

        self.futures = futures

    def select_scenes(self, scenes: List[int]):
        """Select scenes by index. The indices are available in thumbs or `items_df`"""
        self.items_df["Selected"] = False
        self.items_df.loc[scenes, "Selected"] = True

    def __getitem__(self, selector: Union[int, slice, List[int]]):
        """Get item"""
        if isinstance(selector, int):
            return self.items_df.loc[selector, "S2Tile"]
        else:
            return self.items_df.loc[selector, "S2Tile"].to_list()
