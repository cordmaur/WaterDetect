from pathlib import Path
from .Common import DWutils
from . import gdal
from typing import Union


def search_mask(img_path: Path, masks_path: Path, img_type='S2_S2COR') -> Union[Path, None]:
    """
    Given an image and a path for masks, find the correct mask for the image, otherwise return None.
    The match is done by name. The mask name has to contain the images name.
    :param img_path: path to an image
    :param masks_path: path to mask images
    :param img_type: image name convention. Only 'S2_S2COR' is currently supported.
    :return: The mask path or None
    """
    img_detail = DWutils.parse_img_name(img_path.stem, img_type=img_type)

    if img_detail is not None:
        for mask in masks_path.iterdir():
            mask_detail = DWutils.parse_img_name(mask.name, img_type=img_type)

            if (img_detail['mission'] == mask_detail['mission']) and \
               (img_detail['datetime'] == mask_detail['datetime']) and \
               (img_detail['tile'] == mask_detail['tile']):
                return mask

    return


def process_mask(mask: Path, img: Path, flags: list, valid_value=0, invalid_value=255) -> None:

    # read the given mask
    ds = gdal.Open(mask.as_posix())
    mask_array = ds.ReadAsArray()

    # the flags will be set to the invalid value
    for flag in flags:
        mask_array[mask_array == flag] = invalid_value

    # all other points will be valid
    mask_array[mask_array != invalid_value] = valid_value

    # at the end, we will save the mask into the img path
    mask_name = (img/mask.name).as_posix()

    DWutils.array2raster(mask_name, mask_array, ds.GetGeoTransform(), ds.GetProjection(), gdal.GDT_UInt16)


def prepare_external_masks(imgs_dir: str, masks_dir: str, flags: list, img_type: str = 'S2_S2COR') -> None:

    # convert the directories to Paths
    imgs_path, masks_path = Path(imgs_dir), Path(masks_dir)

    # read the images and the masks into lists
    imgs = [f for f in imgs_path.iterdir() if f.is_dir()]

    # loop through the images. For each image, we will look for the corresponding mask
    for img in imgs:
        mask = search_mask(img, masks_path, img_type)
        if mask is not None:
            process_mask(mask, img, flags)
