from pathlib import Path
from .Common import DWutils
from . import gdal
from typing import Union
import numpy as np
from skimage.morphology import binary_dilation


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


def set_mask_flags(mask: np.ndarray, flags: list, dilation: int = 0) -> np.ndarray:
    """
    Given a mask, set the pixels that have flag values as True and others as False.
    Return this boolean array.
    @param mask: path to a geotiff with the mask
    @param flags: values of the mask that shall be set to True
    @param dilation: Applies a dilation on the True pixels. Default=0 (no dilation)
    @return: boolean array
    """

    # create the boolean result
    bool_mask = np.zeros_like(mask).astype('bool')

    # the flags will be set to the invalid value
    for flag in flags:
        bool_mask[mask == flag] = True

    # if a dilation size is set
    if dilation > 0:
        bool_mask = binary_dilation(bool_mask, np.ones((dilation, dilation)))

    return bool_mask


def process_mask(mask: Path, img: Path, flags: list, valid_value=0, invalid_value=255, dilation=0) -> None:
    """
    Given a mask path and the satellite image path, set the pixels with values in the flags list to invalid value
    and remaining pixels to valid_value. At the end, the processed mask will be saved to the image directory.
    @param mask: path to the mask geotiff
    @param img: path to the image directory, where the mask will be copied into.
    @param flags: flags on the mask image that should be masked (set to True)
    @param valid_value: value to be assigned to valid pixels
    @param invalid_value: value to be assigned to invalid pixels
    @param dilation: Size of the kernel (square) to be applied in the dilation processing
    @return: Nothing
    """

    # read the given mask
    ds = gdal.Open(mask.as_posix())
    input_mask = ds.ReadAsArray().astype('uint8')

    # create the binary mask
    binary_mask = set_mask_flags(input_mask, flags, dilation)

    # assign the values accordingly
    mask_array = np.where(binary_mask, invalid_value, valid_value)

    # at the end, we will save the mask into the img path
    mask_name = (img/mask.name).as_posix()

    DWutils.array2raster(mask_name, mask_array, ds.GetGeoTransform(), ds.GetProjection(), gdal.GDT_UInt16)


def prepare_external_masks(imgs_dir: str, masks_dir: str, flags: list, img_type: str = 'S2_S2COR', dilation=0) -> None:

    # convert the directories to Paths
    imgs_path, masks_path = Path(imgs_dir), Path(masks_dir)

    # read the images and the masks into lists
    imgs = [f for f in imgs_path.iterdir() if f.is_dir()]

    # loop through the images. For each image, we will look for the corresponding mask
    for img in imgs:
        try:
            mask = search_mask(img, masks_path, img_type)
            if mask is not None:
                print(f'Processing mask: {str(mask)}')
                process_mask(mask, img, flags, dilation=dilation)
                
        except Exception as e:
            print(e)
            print(f'Problem processing mask {mask.name}')
