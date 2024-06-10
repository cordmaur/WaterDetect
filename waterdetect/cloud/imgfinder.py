"""
Main module that defines the RSImageFinder class
"""

from typing import Optional, List, Type, Union, Iterable

import pystac
from pystac_client import Client
from pystac.extensions.raster import RasterBand
from odc.stac import stac_load, extract_collection_metadata
from odc.geo.geobox import GeoBox
import planetary_computer as pc

import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import shape, box
import geopandas as gpd
import xarray as xr

from tqdm.notebook import tqdm

from .rsimage import RSImage
from .s2 import S2Image


class RSImageFinder:
    """
    The RSImageFinder class is responsible for querying and fetching satellite
    imagery from a STAC API. It provides methods for searching for images in a
    region, and for fetching the images.
    """

    default_catalog = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(
        self, stac_catalog: Optional[str] = None, img_type: Type[RSImage] = S2Image
    ):
        """Init method

        Args:
            stac_catalog (Optional[str], optional): STAC Catalog URL.
            Is None, defaults to "https://planetarycomputer.microsoft.com/api/stac/v1".

            img_type (Type[RSImage], optional): The Image type that will be searched for.
            The type has to be derived from the abstract RSImage class.
            Defaults to S2Image.
        """

        if not issubclass(img_type, RSImage):
            raise TypeError(f"{img_type} is not a subclass of RSImage")

        self.catalog = stac_catalog or RSImageFinder.default_catalog
        self.img_type = img_type

        self.client = Client.open(self.catalog)

        # Initialize state variables
        self.bbox = None
        self._thumbs_cube = None
        self.shape = None
        self.metadata = None
        self.items = None
        self.stac_items = None

    # ---------------------------------------------------------------------
    # State Management Methods
    # ---------------------------------------------------------------------
    def reset_state(self):
        """Reset all state variables"""
        self.bbox = None
        self.shape = None
        self._thumbs_cube = None
        self.metadata = None
        self.items = None
        self.stac_items = None

    def set_state(
        self, items: Iterable[pystac.Item], bbox: Optional[List[float]] = None
    ):
        """Given a list of STAC Items, set the state accordingly"""

        # set the items and metadata
        self.stac_items = items
        self.metadata = extract_collection_metadata(items[0])

        # use the first bands as reference for querying
        band = self.metadata.all_bands[0]

        # get the distinct datetimes, considering the solar_day groupby clause
        ds = stac_load(
            items=items,
            bands=[band],
            resolution=self.img_type.metadata["base_resolution"],
            patch_url=pc.sign,
            chunks={"time": 256, "x": 512, "y": 512},
            nodata=0,
            groupby="solar_day",
            bbox=bbox,
            stac_cfg=self.img_type.stac_cfg,
        )

        # once retrieved the band and grouped by solar day, let's group
        # the items that will compose each RSImage
        # First, we will create a dataframe with the stac items
        self.items = pd.DataFrame(
            {
                "item": items,
                "date": [item.datetime.replace(tzinfo=None) for item in items],
            }
        )
        # Then we create an index with the reference dates
        ref_dates = pd.DatetimeIndex(pd.to_datetime(ds.time))
        self.items["group"] = ref_dates.get_indexer(
            self.items["date"], method="nearest"
        )

        self.items["tiling"] = self.img_type.get_tiling()
        self.items["tile"] = list(map(self.img_type.extract_tile, items))

        # Get the tiles within the search
        self.shape = ds[band].shape[-2:]

        if bbox:
            self.bbox = bbox

        else:
            # if no bbox is passed, than it is assumed we are working with a single tile
            if len(self.tiles) != 1:
                raise ValueError("No bbox provided and multiple tiles found.")

            # Extract the bbox from the geometry
            geom = shape(items[0].geometry)
            self.bbox = list(geom.bounds)

        # print current status
        print(self)

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def dates(self):
        """Get the dates, ignoring the time component"""
        dates = set([date.strftime("%Y-%m-%d") for date in self.items["date"]])
        return list(dates)

    @property
    def thumbs_cube(self):
        """Load a cube for the thumbnails"""
        if not self.bbox:
            raise ValueError("No bbox found. Run `search_region` first.")

        if self._thumbs_cube is None:
            print(f"Loading thumbnails ({self.img_type.__name__})...")

            # Create a geobox with low resolution for each thumb
            geobox = GeoBox.from_bbox(
                bbox=self.bbox,
                crs="epsg:4326",
                shape=256,
            )

            if "visual" in self.metadata.all_bands:
                # This hack is made to create the information about the visual raster bands
                # https://github.com/opendatacube/odc-stac/issues/155
                for item in self.stac_items:
                    item.ext.add("raster")
                    item.assets["visual"].ext.raster.bands = [
                        RasterBand.create(nodata=0, data_type="uint8"),
                        RasterBand.create(nodata=0, data_type="uint8"),
                        RasterBand.create(nodata=0, data_type="uint8"),
                    ]

                rgb_bands = ["visual.1", "visual.2", "visual.3"]
                scale = 0.4 / 255
            else:
                # If no visual bands, let's use the original surface reflectance (RGB) bands
                rgb_bands = ["red", "green", "blue"]
                scale = self.img_type.metadata.get("scale", 1)

            self._thumbs_cube = (
                stac_load(
                    items=self.stac_items,
                    bands=rgb_bands,
                    patch_url=pc.sign,
                    # dtype=dtype,
                    nodata=0,
                    groupby="solar_day",
                    geobox=geobox,
                    progress=tqdm,
                    stac_cfg=self.img_type.stac_cfg,
                )
                .to_array(dim="band")
                .astype("float32")
                * scale
            )

        return self._thumbs_cube

    @property
    def tiles(self):
        """Get unique tiles"""
        return set(self.items["tile"])

    # ---------------------------------------------------------------------
    # Static methods
    # ---------------------------------------------------------------------
    @staticmethod
    def _adjust_borders(axs: List[plt.Axes]):
        # once the images plotted, let's delete the axis
        for _, ax in enumerate(axs):
            if not ax.has_data():
                ax.axis("off")
            else:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # selected = select_list[idx]
                color = "green"  # if selected else "red"

                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(1.5)

    @staticmethod
    def _estimate_cloud_cover(img: xr.DataArray):
        """Estimate the cloud cover for an image"""
        mean = img.mean(dim="band")
        valid_pixels = (mean > 0).sum().data
        cloud_pixels = (mean > 0.3).sum().data
        return cloud_pixels / valid_pixels

    # ---------------------------------------------------------------------
    # Plotting methods
    # ---------------------------------------------------------------------
    def plot_tiling(self):
        """Docstring"""

        if not self.stac_items:
            raise ValueError("No items found. Run `search_region` first.")

        shapes = [shape(item.geometry) for item in self.stac_items]
        gdf = gpd.GeoDataFrame(geometry=shapes)
        ax = gdf.plot(facecolor="red", alpha=0.02)

        gdf.plot(ax=ax, facecolor="none", edgecolor="black")

        gpd.GeoDataFrame(geometry=[box(*self.bbox)]).plot(
            ax=ax, facecolor="blue", alpha=0.5
        )

    def plot_thumbs(self, col_wrap: int = 4, vmax: float = 0.4, cell_size: int = 4):
        """Docstring"""

        if not self.stac_items:
            raise ValueError("No items found. Run `search_region` first.")

        n = len(self.dates)
        rows = n // col_wrap + (1 if n % col_wrap > 0 else 0)

        fig, axs = plt.subplots(
            rows, col_wrap, figsize=(col_wrap * cell_size, rows * cell_size)
        )

        fig.suptitle(f"Thumbnails for {self.dates[0]} to {self.dates[-1]}")

        # Flatten the axes
        axs = axs.flatten()

        for i, date in enumerate(self.dates):
            ax = axs[i]
            img = self.thumbs_cube.isel(time=i)

            # Plot the image
            if RSImageFinder._estimate_cloud_cover(img) > 0.7:
                img.plot.imshow(ax=ax, rgb="band", robust=True)
            else:
                img.plot.imshow(ax=ax, rgb="band", vmax=vmax, robust=True)

            ax.set_title(f"{i}: {date}")

        RSImageFinder._adjust_borders(axs)

    # ---------------------------------------------------------------------
    # Main methods
    # ---------------------------------------------------------------------
    def search_region(self, bbox: List[float], time_range: str):
        """Search for images in a region

        Args:
            bbox (List[float]): xmin, ymin, xmax, ymax
            time_range (str): Either a date-time or an interval, open or closed.
                Date and time expressions adhere to RFC 3339. Open uses double-dots.
                Examples:
                A date-time: "2018-02-12T23:20:50Z"
                A closed interval: "2018-02-12T00:00:00Z/2018-03-18T12:31:12Z"
                Open intervals: "2018-02-12T00:00:00Z/.." or "../2018-03-18T12:31:12Z"
        """

        # Before a new search, reset the state variables
        self.reset_state()

        print(f"Searching for items on {self.client.links[0].href}")
        search = self.client.search(
            collections=[self.img_type.collection],
            bbox=bbox,
            datetime=time_range,
        )

        self.set_state(search.item_collection(), bbox=bbox)
        # # get the items
        # items = search.item_collection()
        # self.item_collection = items
        # self.metadata = extract_collection_metadata(items[0])

        # # use the first bands as reference for querying
        # band = self.metadata.all_bands[0]

        # # get the distinct datetimes, considering the solar_day groupby clause
        # ds = stac_load(
        #     items=items,
        #     bands=[band],
        #     resolution=self.img_type.metadata["base_resolution"],
        #     patch_url=pc.sign,
        #     chunks={"time": 256, "x": 512, "y": 512},
        #     nodata=0,
        #     groupby="solar_day",
        #     stac_cfg=self.img_type.stac_cfg,
        #     bbox=bbox,
        # )

        # # once retrieved the band and grouped by solar day, let's group
        # # the items that will compose each RSImage
        # # First, we will create a dataframe with the stac items
        # self.items = pd.DataFrame(
        #     {
        #         "item": self.item_collection,
        #         "date": [item.datetime.replace(tzinfo=None) for item in items],
        #     }
        # )
        # # Then we create an index with the reference dates
        # ref_dates = pd.DatetimeIndex(pd.to_datetime(ds.time))
        # self.items["group"] = ref_dates.get_indexer(
        #     self.items["date"], method="nearest"
        # )

        # self.items["tiling"] = self.img_type.get_tiling()
        # self.items["tile"] = list(map(self.img_type.extract_tile, items))

        # # Get the tiles within the search
        # self.datetimes = [pd.to_datetime(d).to_pydatetime() for d in ds.time.values]
        # self.shape = ds[band].shape[-2:]
        # self.bbox = bbox

        # # print current status
        # print(self)

        # return ds

    def search_tile(
        self,
        tile: Union[str, List[str]],
        time_range: str,
        max_cloud: int = 100,
    ):
        """Search for the tile in the desired time range.

        Args:
            tile (Union[str, List[str]]): Tile to be searched for. If the tile is specified by
            more than one property (e.g., path and row in Landsat), they should be provided in
            a list, following the order in `img_type.metadata['tile_property']`
            time_range (str): Either a date-time or an interval, open or closed.
                Date and time expressions adhere to RFC 3339. Open uses double-dots.
                Examples:
                A date-time: "2018-02-12T23:20:50Z"
                A closed interval: "2018-02-12T00:00:00Z/2018-03-18T12:31:12Z"
                Open intervals: "2018-02-12T00:00:00Z/.." or "../2018-03-18T12:31:12Z"

            max_cloud (int, optional): Maximum cloud cover percentage of the tile. Defaults to 100.
        """
        # Before a new search, reset the state variables
        self.reset_state()

        query = {"eo:cloud_cover": {"lt": max_cloud}}

        # check if the tile is defined by more than one property
        if isinstance(self.img_type.metadata["tile_property"], list):
            properties = self.img_type.metadata["tile_property"]
            if len(properties) != len(tile):
                raise ValueError(
                    f"The number of properties in the tile definition ({len(properties)}) "
                    f"does not match the number of values provided ({len(tile)})"
                )
            for prop, value in zip(properties, tile):
                query[prop] = {"eq": value}
        else:
            query[self.img_type.metadata["tile_property"]] = {"eq": tile}

        # search the catalog for the tiles within the date range
        search = self.client.search(
            collections=[self.img_type.collection],
            query=query,
            datetime=time_range,
        )

        self.set_state(search.item_collection())

    # ---------------------------------------------------------------------
    # Dunder methods
    # ---------------------------------------------------------------------
    def __repr__(self):

        s = f"Collection: {self.img_type.collection}\n"

        if self.items is None:
            s += "Empty object. Run `search_` to find images."

        else:
            s = f"{len(self.stac_items)} STAC items loaded in {len(self.tiles)} tiles and "
            s += f"{len(self.dates)} solar dates\n"
            s += f"{self.img_type.get_tiling()}: {self.tiles}\n"
            s += f"Output shape {self.shape}, "
            s += f"considering default resolution: {self.img_type.metadata['base_resolution']}"

        return s


# class old_RSImageFinder:
#     """Docstring"""

#     default_catalog = "https://planetarycomputer.microsoft.com/api/stac/v1"
#     default_collection = "sentinel-2-l2a"
#     resolution = 10

#     def __init__(
#         self, stac_catalog: Optional[str] = None, collection: Optional[str] = None
#     ):
#         """Docstring"""
#         self.catalog = stac_catalog or RSImageFinder.default_catalog
#         self.collection = collection or RSImageFinder.default_collection

#         self.client = Client.open(self.catalog)

#         # Initialize state variables
#         self.bbox = None
#         self.tiles = None
#         self.datetimes = None
#         self.item_collection = None
#         self._thumbs_cube = None
#         self.shape = None
#         self.metadata = None

#     # ---------------------------------------------------------------------
#     # State Management Methods
#     # ---------------------------------------------------------------------
#     def reset_state(self):
#         """Reset all state variables"""
#         self.bbox = self.tiles = self.datetimes = None
#         self.item_collection = self.shape = None
#         self._thumbs_cube = None
#         self.metadata = None

#     # ---------------------------------------------------------------------
#     # Properties
#     # ---------------------------------------------------------------------
#     @property
#     def dates(self):
#         """Get the dates, ignoring the time component"""
#         return [date.strftime("%Y-%m-%d") for date in self.datetimes]

#     @property
#     def thumbs_cube(self):
#         """Load a cube for the thumbnails"""
#         if not self.bbox:
#             raise ValueError("No bbox found. Run `search_region` first.")

#         if self._thumbs_cube is None:
#             print("Loading thumbnails...")

#             # Create a geobox with low resolution for each thumb
#             geobox = GeoBox.from_bbox(
#                 bbox=self.bbox,
#                 crs="epsg:4326",
#                 shape=256,
#             )

#             if "visual" in self.metadata.all_bands:
#                 # This hack is made to create the information about the visual raster bands
#                 # https://github.com/opendatacube/odc-stac/issues/155
#                 for item in self.item_collection:
#                     item.ext.add("raster")
#                     item.assets["visual"].ext.raster.bands = [
#                         RasterBand.create(nodata=0, data_type="uint8"),
#                         RasterBand.create(nodata=0, data_type="uint8"),
#                         RasterBand.create(nodata=0, data_type="uint8"),
#                     ]

#                 rgb_bands = ["visual.1", "visual.2", "visual.3"]
#                 dtype = None
#                 scale = 0.4 / 255
#             else:
#                 rgb_bands = ["red", "green", "blue"]
#                 dtype = "float16"
#                 scale = 1

#             self._thumbs_cube = (
#                 stac_load(
#                     items=self.item_collection,
#                     bands=rgb_bands,
#                     patch_url=pc.sign,
#                     dtype=dtype,
#                     nodata=0,
#                     groupby="solar_day",
#                     geobox=geobox,
#                     progress=tqdm,
#                 )
#                 .to_array(dim="band")
#                 .astype("float16")
#                 * scale
#             )

#         return self._thumbs_cube

#     # ---------------------------------------------------------------------
#     # Static methods
#     # ---------------------------------------------------------------------
#     @staticmethod
#     def _adjust_borders(axs: List[plt.Axes]):
#         # once the images plotted, let's delete the axis
#         for _, ax in enumerate(axs):
#             if not ax.has_data():
#                 ax.axis("off")
#             else:
#                 ax.get_xaxis().set_visible(False)
#                 ax.get_yaxis().set_visible(False)

#                 # selected = select_list[idx]
#                 color = "green"  # if selected else "red"

#                 for spine in ax.spines.values():
#                     spine.set_color(color)
#                     spine.set_linewidth(1.5)

#     @staticmethod
#     def _estimate_cloud_cover(img: xr.DataArray):
#         """Estimate the cloud cover for an image"""
#         mean = img.sum(dim="band") / 3
#         valid_pixels = (mean > 0).sum().data
#         cloud_pixels = (mean > 0.3).sum().data
#         return cloud_pixels / valid_pixels

#     # ---------------------------------------------------------------------
#     # Plotting methods
#     # ---------------------------------------------------------------------
#     def plot_tiling(self):
#         """Docstring"""

#         if not self.item_collection:
#             raise ValueError("No items found. Run `search_region` first.")

#         shapes = [shape(item.geometry) for item in self.item_collection]
#         gdf = gpd.GeoDataFrame(geometry=shapes)
#         ax = gdf.plot(facecolor="red", alpha=0.02)

#         gdf.plot(ax=ax, facecolor="none", edgecolor="black")

#         gpd.GeoDataFrame(geometry=[box(*self.bbox)]).plot(
#             ax=ax, facecolor="blue", alpha=0.5
#         )

#     def plot_thumbs(self, col_wrap: int = 4, vmax: float = 0.3, cell_size: int = 4):
#         """Docstring"""

#         if not self.item_collection:
#             raise ValueError("No items found. Run `search_region` first.")

#         n = len(self.dates)
#         rows = n // col_wrap + (1 if n % col_wrap > 0 else 0)

#         fig, axs = plt.subplots(
#             rows, col_wrap, figsize=(col_wrap * cell_size, rows * cell_size)
#         )

#         fig.suptitle(f"Thumbnails for {self.dates[0]} to {self.dates[-1]}")

#         # Flatten the axes
#         axs = axs.flatten()

#         for i, date in enumerate(self.dates):
#             ax = axs[i]
#             img = self.thumbs_cube.isel(time=i)

#             # Plot the image
#             if RSImageFinder._estimate_cloud_cover(img) > 0.7:
#                 img.plot.imshow(ax=ax, rgb="band", robust=True)
#             else:
#                 img.plot.imshow(ax=ax, rgb="band", vmax=vmax)

#             ax.set_title(f"{i}: {date}")

#         RSImageFinder._adjust_borders(axs)

#     # ---------------------------------------------------------------------
#     # Main methods
#     # ---------------------------------------------------------------------
#     def search_region(self, bbox: List[float], time_range: str):
#         """Search for images in a region

#         Args:
#             bbox (List[float]): xmin, ymin, xmax, ymax
#             time_range (str): Either a date-time or an interval, open or closed.
#                 Date and time expressions adhere to RFC 3339. Open uses double-dots.
#                 Examples:
#                 A date-time: "2018-02-12T23:20:50Z"
#                 A closed interval: "2018-02-12T00:00:00Z/2018-03-18T12:31:12Z"
#                 Open intervals: "2018-02-12T00:00:00Z/.." or "../2018-03-18T12:31:12Z"
#         """

#         # Before a new search, reset the state variables
#         self.reset_state()

#         print(f"Searching for items on {self.client.links[0].href}")
#         search = self.client.search(
#             collections=[self.collection],
#             bbox=bbox,
#             datetime=time_range,
#         )

#         # get the items
#         items = search.item_collection()
#         self.item_collection = items
#         self.metadata = extract_collection_metadata(items[0])
#         self.tiles = set(map(lambda item: item.properties["s2:mgrs_tile"], items))

#         # use the first band as reference for querying
#         band = self.metadata.all_bands[0]

#         # get the distinct datetimes, considering the solar_day groupby clause
#         ds = stac_load(
#             items=items,
#             bands=[band],
#             resolution=10,
#             patch_url=pc.sign,
#             chunks={"time": 256, "x": 512, "y": 512},
#             dtype="uint16",
#             nodata=0,
#             groupby="solar_day",
#             bbox=bbox,
#         )

#         self.datetimes = [pd.to_datetime(d).to_pydatetime() for d in ds.time.values]
#         self.shape = ds[band].shape[-2:]
#         self.bbox = bbox

#         print(
#             f"{len(items)} STAC items found in {len(self.tiles)} tiles and "
#             f"{len(self.datetimes)} dates"
#         )
#         print(self.tiles)

#         print(
#             f"Output shape {self.shape}, "
#             f"considering default resolution: {self.resolution}"
#         )

#     def search_tiles(
#         self,
#         tile: str,
#         time_range: str,
#         max_cloud: int = 100,
#     ):
#         """Search for tiles

#         Args:
#             tile (str): _description_
#             time_range (str): _description_
#             max_cloud (int, optional): _description_. Defaults to 100.

#         Returns:
#             _type_: _description_
#         """
#         query = {"eo:cloud_cover": {"lt": max_cloud}, "s2:mgrs_tile": {"eq": tile}}

#         search = self.client.search(
#             collections=[self.default_collection],
#             query=query,
#             datetime=time_range,
#         )

#         return search.get_items
