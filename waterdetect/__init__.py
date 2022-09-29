# todo: Implement logging
# import logging
__version__ = '1.5.13'


class DWProducts:
    Landsat8_USGS = 'L8_USGS'
    Sentinel2_THEIA = 'S2_THEIA'
    Sentinel2_ESA = 'S2_S2COR'
    Sentinel2_L1C = 'S2_L1C'

try:
    from osgeo import gdal
    # just imports DWWaterDetect if gdal is present

except BaseException as error:
    # print(error)
    print('GDAL not found in environment. Waterdetect can still run as API calling DWImageClustering and passing'
          ' the arrays as dictionary. Refer to online documentation. No call to DWWaterDetect, that requires '
          'loading satellite images from disk will be possible')
    gdal = None

# Correct the jaccard score name depending on the sklearn version
from packaging import version
from sklearn import __version__ as skversion

if version.parse(skversion) < version.parse('0.21'):
    from sklearn.metrics import jaccard_similarity_score as jaccard_score
    from sklearn.metrics import calinski_harabaz_score as calinski_harabasz_score
else:
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import calinski_harabasz_score

from waterdetect.WaterDetect import DWWaterDetect
from waterdetect.Image import DWImageClustering
from waterdetect.Common import DWutils, DWConfig
from waterdetect.External_mask import prepare_external_masks


