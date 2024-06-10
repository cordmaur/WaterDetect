
from .rsimage import RSImage

class L8Image(RSImage):
    """_summary_

    Args:
        RSImage (_type_): _description_
    """

    catalog = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection = "landsat-c2-l2"
    metadata = {
        "base_resolution": 30,
        "scale": 0.0000275,
        "tiling": "Path/Row",
        "tile_property": ["landsat:wrs_path", "landsat:wrs_row"],
    }
