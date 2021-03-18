# todo: Implement logging
# import logging
__version__ = '1.5.5'

class DWProducts:
    Landsat8_USGS = 'L8_USGS'
    Sentinel2_THEIA = 'S2_THEIA'
    Sentinel2_ESA = 'S2_S2COR'
    Sentinel2_L1C = 'S2_L1C'


try:
    from osgeo import gdal
    # just imports DWWaterDetect if gdal is present
    from waterdetect.WaterDetect import DWWaterDetect

except BaseException as error:
    # print(error)
    print('GDAL not found in environment. Waterdetect can still run as API calling DWImageClustering and passing'
          ' the arrays as dictionary. Refer to online documentation. No call to DWWaterDetect, that requires '
          'loading satellite images from disk will be possible')
    gdal = None

from waterdetect.Image import DWImageClustering
from waterdetect.Common import DWutils, DWConfig

