"""
This module manages all the access to the Planetary Computer through the 
PCDownloader class
"""

from typing import Tuple, List, Optional

import numpy as np

import xarray as xr
import rasterio as rio
import rioxarray as xrio

import planetary_computer as pc
import pystac
import pystac_client

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class PCDownloader:
    """PCDownloader class"""

    def __init__(self, catalog_url: str = CATALOG_URL):
        self.catalog = pystac_client.Client.open(catalog_url)

    def search(self, **kwargs) -> pystac.ItemCollection:
        """Return items matching the search criteria.

        Args:
            **kwargs: Must conform to the STAC search API.
            https://api.stacspec.org/v1.0.0-beta.3/item-search/#tag/Item-Search/operation/getItemSearch

        Returns:
            pystac.ItemCollection: The items matching the search criteria.
        """
        print("Searching for items...")
        items = self.catalog.search(**kwargs).item_collection()
        print(f"{len(items)} items found.")

        return items

    @staticmethod
    def get_asset(
        item: pystac.Item,
        asset: str,
        shape: Optional[Tuple[int, int]] = None,
        scale: float = 1e-4,
    ) -> xr.DataArray:
        """Retrieve an asset from a pystac.item.

        Args:
        item (pystac.Item): The item to retrieve the asset from.
        asset (str): The name of the asset to retrieve.
        shape (Optional[Tuple[int, int]]): The desired shape of the asset as a tuple
            of height width and width . If not provided the asset will be returned at its
            original resolution.
        scale (float): The scale factor to apply to the asset. Defaults to 1e-4.

        Returns:
        The requested asset as an XArray.DataArray.

        Examples:
        Retrieve a thumbnail (scaled down) GeoTiff asset:

        ```python
        asset = get_asset(item, 'B02', scale=0.1)
        """

        # Open the asset
        href = pc.sign(item.assets[asset]).href
        ds = rio.open(href)

        # Read it with the desired output shape (decimated reading)
        data = ds.read(1, out_shape=shape)

        # Replace nodata values with NaN
        if ds.nodata:
            data[data == ds.nodata] = np.nan

        data = data * scale

        # create the lats and longs coordinates, considering the bounds and the shape
        longs = np.linspace(ds.bounds.left, ds.bounds.right, shape[1])
        lats = np.linspace(ds.bounds.top, ds.bounds.bottom, shape[0])

        # create the DataArray
        arr = xr.DataArray(data=data, dims=["y", "x"], coords=[lats, longs])

        # set CRS and GeoTransform
        arr = arr.rio.write_crs(ds.crs)
        transform = rio.transform.from_bounds(
            *ds.bounds, height=shape[0], width=shape[1]
        )

        arr = arr.rio.write_transform(transform)

        return arr

    @staticmethod
    def get_assets(
        item: pystac.Item, assets: List, shape: Tuple[int, int], scale: float = 1e-4
    ) -> xr.DataArray:
        """
        Retrieve multiple assets from a STAC item.

        Args:
            item (pystac.Item): The STAC item to retrieve assets from.
            assets (List[str]): A list of asset names to retrieve.
            shape (Optional[Tuple[int,int]]): The desired shape as a width,height tuple.
            If None the original resolution is used.
            scale (float): The scale factor to apply to all assets.

        Returns:
            xarray.DataArray: A DataArray with the assets stacked on the band dimension.
            The band dimension contains the names of the retrieved assets.
            The coordinates are the latitudes and longitudes of the
            retrieved assets as variables.

        Examples:

            Get the red and nir bands from a Landsat scene as a xarray dataset:

            ```python
            ds = get_assets(item, ['red', 'nir'])
            ```
        """
        # Retrieve the assets
        arrs = {a: PCDownloader.get_asset(item, a, shape, scale) for a in assets}

        # Concatenate the assets into a single xarray DataArray
        cube = xr.concat(arrs.values(), dim="band")
        cube["band"] = list(arrs.keys())
        return cube
