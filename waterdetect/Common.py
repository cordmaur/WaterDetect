import os
from shutil import copy
import configparser
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from waterdetect import DWProducts, gdal

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lxml import etree
from PIL import Image, ImageDraw, ImageFont


def test_ini():
    for attr in dir(DWProducts):
        print(getattr(DWProducts, attr))


class DWConfig:

    _config_file = 'WaterDetect.ini'
    _defaults = {'reference_band': 'Red',
                 'maximum_invalid': '0.8',
                 'create_composite': 'True',
                 'pdf_reports': 'False',
                 'save_indices': 'False',
                 'texture_streching': 'False',
                 'clustering_method': 'aglomerative',
                 'min_clusters': '1',
                 'max_clusters': '5',
                 'clip_band': 'None',
                 'clip_value': 'None',
                 'classifier': 'naive_bayes',
                 'train_size': '0.1',
                 'min_train_size': '1000',
                 'max_train_size': '10000',
                 'score_index': 'calinsk',
                 'detectwatercluster': 'maxmndwi',
                 'clustering_bands': "[['ndwi', 'Nir']]",
                 'graphs_bands': "[['mbwi', 'mndwi'], ['ndwi', 'mbwi']]",
                 'plot_ts': 'False',
                 'calc_glint': 'True'
                 }

    _units = {'turb-dogliotti': 'FNU',
              'spm-get': 'mg/l',
              'chl-lins': 'mg/m^3',
              'chl-giteslon': 'mg/m^3',
              'aCDOM-brezonik': 'Absorption Coef'}

    def __init__(self, config_file=None):

        self.config = self.load_config_file(config_file)

        return

    def return_defaults(self, section, key):

        default_value = self._defaults[key]

        print('Key {} not found in section {}: using default value {}'.format(key, section, default_value))

        return default_value

    def get_option(self, section, key, evaluate: bool):

        try:
            str_value = self.config.get(section, key)

        except Exception as err:
            str_value = self.return_defaults(section, key)

        if evaluate and str == type(str_value):
            try:
                return ast.literal_eval(str_value)
            except Exception as err:
                return str_value
        else:
            return str_value

    def load_config_file(self, config_file):

        if config_file:
            self._config_file = config_file

        print('Loading configuration file {}'.format(self._config_file))

        DWutils.check_path(self._config_file)

        config = configparser.ConfigParser()
        config.read(self._config_file)

        return config

    @property
    def reference_band(self):
        return self.get_option('General', 'reference_band', evaluate=False)

    @property
    def create_composite(self):
        return self.get_option('General', 'create_composite', evaluate=True)

    @property
    def pdf_reports(self):
        return self.get_option('General', 'pdf_reports', evaluate=True)

    @property
    def calc_glint(self):
        return self.get_option('General', 'calc_glint', evaluate=True)

    @property
    def glint_mode(self):
        return self.get_option('General', 'glint_mode', evaluate=True)

    @property
    def min_glint_multiplier(self):
        return self.get_option('General', 'min_glint_multiplier', evaluate=True)

    @property
    def pdf_resolution(self):
        return self.get_option('General', 'pdf_resolution', evaluate=True)

    @property
    def pekel_water(self):
        return self.get_option('General', 'pekel_water', evaluate=True)

    @property
    def pekel_accuracy(self):
        return self.get_option('General', 'pekel_accuracy', evaluate=True)

    @property
    def save_indices(self):
        return self.get_option('General', 'save_indices', evaluate=True)

    @property
    def texture_stretching(self):
        return self.get_option('General', 'texture_stretching', evaluate=True)

    @property
    def external_mask(self):
        return self.get_option('External_Mask', 'external_mask', evaluate=True)

    @property
    def mask_name(self):
        return self.get_option('External_Mask', 'mask_name', evaluate=False)

    @property
    def mask_valid_value(self):
        return self.get_option('External_Mask', 'mask_valid_value', evaluate=True)

    @property
    def mask_invalid_value(self):
        return self.get_option('External_Mask', 'mask_invalid_value', evaluate=True)

    @property
    def inversion(self):
        return self.get_option('Inversion', 'inversion', evaluate=True)

    @property
    def parameter(self):
        if self.inversion:
            return self.get_option('Inversion', 'parameter', evaluate=False)
        else:
            return ''

    @property
    def parameter_unit(self):

        return self._units[self.parameter]

    @property
    def negative_values(self):
        return self.get_option('Inversion', 'negative_values', evaluate=False)

    @property
    def min_param_value(self):
        return self.get_option('Inversion', 'min_param_value', evaluate=True)

    @property
    def max_param_value(self):
        return self.get_option('Inversion', 'max_param_value', evaluate=True)

    @property
    def colormap(self):
        return self.get_option('Inversion', 'colormap', evaluate=False)

    @property
    def colormap(self):
        return self.get_option('Inversion', 'colormap', evaluate=False)

    @property
    def uniform_distribution(self):
        return self.get_option('Inversion', 'uniform_distribution', evaluate=True)

    @property
    def maximum_invalid(self):
        return self.get_option('General', 'maximum_invalid', evaluate=True)

    @property
    def average_results(self):
        return self.get_option('Clustering', 'average_results', evaluate=True)

    @property
    def min_positive_pixels(self):
        return self.get_option('Clustering', 'min_positive_pixels', evaluate=True)

    @property
    def clustering_method(self):
        return self.get_option('Clustering', 'clustering_method', evaluate=False)

    @property
    def linkage(self):
        return self.get_option('Clustering', 'linkage', evaluate=False)

    @property
    def train_size(self):
        return self.get_option('Clustering', 'train_size', evaluate=True)

    @property
    def regularization(self):
        return self.get_option('Clustering', 'regularization', evaluate=True)

    @property
    def min_train_size(self):
        return self.get_option('Clustering', 'min_train_size',evaluate=True)

    @property
    def max_train_size(self):
        return self.get_option('Clustering', 'max_train_size', evaluate=True)

    @property
    def score_index(self):
        return self.get_option('Clustering', 'score_index', evaluate=False)

    @property
    def classifier(self):
        return self.get_option('Clustering', 'classifier', evaluate=False)

    @property
    def detect_water_cluster(self):
        return self.get_option('Clustering', 'detectwatercluster', evaluate=False)

    @property
    def min_clusters(self):
        return self.get_option('Clustering', 'min_clusters', evaluate=True)

    @property
    def max_clusters(self):
        return self.get_option('Clustering', 'max_clusters', evaluate=True)

    @property
    def plot_graphs(self):
        return self.get_option('Graphs', 'plot_graphs', evaluate=True)

    @property
    def graphs_bands(self):

        bands_lst = self.get_option('Graphs', 'graphs_bands', evaluate=True)

        # if bands_keys is not a list of lists, transform it
        if type(bands_lst[0]) == str:
            bands_lst = [bands_lst]

        return bands_lst

    @property
    def plot_ts(self):
        return self.get_option('TimeSeries', 'plot_ts', evaluate=True)

    @property
    def clustering_bands(self):

        bands_lst = self.get_option('Clustering', 'clustering_bands', evaluate=True)

        # if bands_keys is not a list of lists, transform it
        if type(bands_lst[0]) == str:
            bands_lst = [bands_lst]

        return bands_lst

    @property
    def clip_band(self):
        bands_lst = self.get_option('Clustering', 'clip_band', evaluate=True)

        if type(bands_lst) == str:
                return [bands_lst]
        else:
            return bands_lst if bands_lst is not None else []

    @property
    def clip_inf_value(self):
        value = self.get_option('Clustering', 'clip_inf_value', evaluate=True)

        if value is not None:
            return value if type(value) is list else [value]
        else:
            return []

    @property
    def clip_sup_value(self):
        value = self.get_option('Clustering', 'clip_sup_value', evaluate=True)

        if value is not None:
            return value if type(value) is list else [value]
        else:
            return []


    def get_masks_list(self, product):

        masks_lst = []

        if product == 'LANDSAT8':
            section_name = 'LandsatMasks'
        elif product == 'S2_THEIA':
            section_name = 'TheiaMasks'
        elif product == 'S2_S2COR':
            section_name = 'S2CORMasks'
        else:
            section_name = None

        if section_name is not None:
            for key in self.config._sections[section_name]:
                if self.config.getboolean(section_name, key):
                    masks_lst.append(key)

        return masks_lst


class DWutils:

    @staticmethod
    def parse_img_name(name, img_type='S2_S2COR'):

        # ignore extension
        name = name.split('.')[0]

        if img_type == 'S2_S2COR':
            lst = name.split('_')
            return dict(mission=lst[0],
                        level=lst[1],
                        datetime=lst[2],
                        pdgs=lst[3],
                        orbit=lst[4],
                        tile=lst[5])

        else:
            print(f'Image type {img_type} is not supported')
            return None

    @staticmethod
    def bitwise_or(array, bit_values):
        return np.bitwise_or(array, bit_values)

    @staticmethod
    def bitwise_and(array, bit_values):
        return np.bitwise_and(array, bit_values)

    @staticmethod
    def listify(lst, uniques=[]):
        # pdb.set_trace()
        for item in lst:
            if isinstance(item, list):
                uniques = DWutils.listify(item, uniques)
            else:
                uniques.append(item)
        return uniques.copy()

    @staticmethod
    def check_path(path_str, is_dir=False):
        """
        Check if the path/file exists and returns a Path variable with it
        :param path_str: path string to test
        :param is_dir: whether if it is a directory or a file
        :return: Path type variable
        """

        if path_str is None:
            return None

        path = Path(path_str)

        if is_dir:
            if not path.is_dir():
                raise OSError('The specified folder {} does not exist'.format(path_str))
        else:
            if not path.exists():
                raise OSError('The specified file {} does not exist'.format(path_str))

        print(('Folder' if is_dir else 'File') + ' {} verified.'.format(path_str))
        return path

    @staticmethod
    def get_directories(input_folder):
        """
        Return a list of directories in input_folder. These folders are the repository for satellite products
        :param input_folder: folder that stores the images
        :return: list of images (i.e. directories)
        """
        return [i for i in input_folder.iterdir() if i.is_dir()]

    @staticmethod
    def calc_normalized_difference(img1, img2, mask=None, compress_cte=0.02):
        """
        Calc the normalized difference of given arrays (img1 - img2)/(img1 + img2).
        Updates the mask if any invalid numbers (ex. np.inf or np.nan) are encountered
        :param img1: first array
        :param img2: second array
        :param mask: initial mask, that will be updated
        :param compress_cte: amount of index compression. The greater, the more the index will be compressed towards 0
        :return: nd array filled with -9999 in the mask and the mask itself
        """

        # changement for negative SRE scenes
        # ##### UPDATED ON 01/04/2021

        # create a minimum array
        min_values = np.where(img1 < img2, img1, img2)

        # then create the to_add matrix (min values turned into positive + epsilon)
        min_values = np.where(min_values <= 0, -min_values + 0.001, 0) + compress_cte

        nd = ((img1 + min_values) - (img2 + min_values)) / ((img1 + min_values) + (img2 + min_values))

        # # VERSION WITH JUST 1 CONSTANT
        # if mask is not None:
        #     min_cte = np.min([np.min(img1[~mask]), np.min(img2[~mask])])
        # else:
        #     min_cte = np.min([np.min(img1), np.min(img2)])
        #
        # print(f'Correcting negative values by {min_cte}')
        #
        # if min_cte <= 0:
        #     min_cte = -min_cte + 0.001
        # else:
        #     min_cte = 0
        #
        # nd = ((img1+min_cte)-(img2+min_cte)) / ((img1+min_cte) + (img2+min_cte))

        # if any of the bands is set to zero in the pixel, makes a small shift upwards, as proposed by olivier hagole
        # https://github.com/olivierhagolle/modified_NDVI
        # nd = np.where((img1 > 0) & (img2 > 0), (img1-img2) / (img1 + img2), np.nan)
        # (img1+0.005-img2-0.005) / (img1+0.005 + img2+0.005))

        # nd = np.where((img1 <= 0) & (img2 <= 0), np.nan, (img1-img2) / (img1 + img2))

        # nd = (img1-img2) / (img1 + img2)

        # nd[~mask] = MinMaxScaler(feature_range=(-1,1), copy=False).fit_transform(nd[~mask].reshape(-1,1)).reshape(-1)

        nd[nd > 1] = 1
        nd[nd < -1] = -1

        # if result is infinite, result should be 1
        nd[np.isinf(nd)] = 1

        # nd_mask = np.isinf(nd) | np.isnan(nd) | mask
        nd_mask = np.isnan(nd) | (mask if mask is not None else False)

        nd = np.ma.array(nd, mask=nd_mask, fill_value=-9999)

        return nd.filled(), nd.mask

    @staticmethod
    def calc_mbwi(bands, factor, mask):
        # changement for negative SRE values scene
        min_cte = np.min([np.min(bands['Green'][~mask]), np.min(bands['Red'][~mask]),
                          np.min(bands['Nir'][~mask]), np.min(bands['Mir'][~mask]), np.min(bands['Mir2'][~mask])])
        if min_cte <= 0:
            min_cte = -min_cte + 0.001
        else:
            min_cte = 0
        mbwi = factor * (bands['Green'] + min_cte) - (bands['Red'] + min_cte) - (bands['Nir'] + min_cte) \
               - (bands['Mir'] + min_cte) - (bands['Mir2'] + min_cte)
        mbwi[~mask] = RobustScaler(copy=False).fit_transform(mbwi[~mask].reshape(-1, 1)).reshape(-1)
        mbwi[~mask] = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(mbwi[~mask].reshape(-1, 1)) \
            .reshape(-1)
        mask = np.isinf(mbwi) | np.isnan(mbwi) | mask
        mbwi = np.ma.array(mbwi, mask=mask, fill_value=-9999)
        return mbwi, mask

    @staticmethod
    def rgb_burn_in(red, green, blue, burn_in_array, color=None, min_value=None, max_value=None, colormap='viridis',
                    fade=1, uniform_distribution=False, no_data_value=-9999, valid_value=1, transp=0.0):
        """
        Burn in a mask or a specific parameter into an RGB image for visualization purposes.
        The burn_in_array will be copied where values are different from no_data_value.
        :param uniform_distribution: convert the input values in a uniform histogram
        :param colormap: matplotlib colormap (string) to create the RGB ramp
        :param max_value: maximum value
        :param min_value: minimum value
        :param red: Original red band
        :param green: Original green band
        :param blue: Original blue band
        :param burn_in_array: Values to be burnt in
        :param no_data_value: Value to ne unconsidered
        :param color: Tuple of color (R, G, B) to be used in the burn in
        :param fade: Fade the RGB bands to emphasize the copied values
        :param transp: Transparency to use in the mask (0=opaque 1=completely transparent)
        :return: RGB image bands
        """

        if color:
            new_red = np.where(burn_in_array == valid_value, color[0] * (1 - transp) + red * (transp), red * fade)
            new_green = np.where(burn_in_array == valid_value, color[1] * (1 - transp) + green * (transp), green * fade)
            new_blue = np.where(burn_in_array == valid_value, color[2] * (1 - transp) + blue * (transp), blue * fade)

        else:
            # the mask is where the value equals no_data_value
            mask = (burn_in_array == no_data_value)

            # the valid values are those outside the mask (~mask)
            burn_in_values = burn_in_array[~mask]

            # apply scalers to uniform the data
            if uniform_distribution:
                burn_in_values = QuantileTransformer().fit_transform(burn_in_values[:, np.newaxis])[:, 0]
            # burn_in_values = MinMaxScaler((0, 0.3)).fit_transform(burn_in_values)

            # rgb_burn_in_values = DWutils.gray2color_ramp(burn_in_values[:, 0], limits=(0, 0.3))
            rgb_burn_in_values = DWutils.gray2color_ramp(burn_in_values, min_value=min_value, max_value=max_value,
                                                         colormap=colormap, limits=(0, 0.25))

            # return the scaled values to the burn_in_array
            # burn_in_array[~mask] = burn_in_values[:, 0]

            # calculate a color_ramp for these pixels
            # rgb_burn_in_values = DWutils.gray2color_ramp(burn_in_array, limits=(0, 0.3))

            # new_red = np.where(burn_in_array == no_data_value, red, rgb_burn_in_values[:, 0])
            # new_green = np.where(burn_in_array == no_data_value, green, rgb_burn_in_values[:, 1])
            # new_blue = np.where(burn_in_array == no_data_value, blue, rgb_burn_in_values[:, 2])

            # return the scaled values to the burn_in_array
            burn_in_array[~mask] = rgb_burn_in_values[:, 0]
            burn_in_red = np.copy(burn_in_array)

            burn_in_array[~mask] = rgb_burn_in_values[:, 1]
            burn_in_green = np.copy(burn_in_array)

            burn_in_array[~mask] = rgb_burn_in_values[:, 2]
            burn_in_blue = np.copy(burn_in_array)

            # burn in the values
            new_red = np.where(burn_in_array == no_data_value, red*fade, burn_in_red)
            new_green = np.where(burn_in_array == no_data_value, green*fade, burn_in_green)
            new_blue = np.where(burn_in_array == no_data_value, blue*fade, burn_in_blue)

        return new_red, new_green, new_blue

    @staticmethod
    def apply_mask(array, mask, no_data_value=-9999, clear_nan=True):

        if clear_nan:
            mask |= np.isnan(array) | np.isinf(array)

        return np.where(mask == True, -9999, array)

    @staticmethod
    def gray2color_ramp(grey_array, color1=(0., 0.0, .6), color2=(0.0, 0.8, 0.), color3=(1., 0., 0.),
                        min_value=0, max_value=20, colormap='viridis', limits=(0, 1)):
        """
        Convert a greyscale n-dimensional matrix into a rgb matrix, adding 3 dimensions to it for R, G, and B
        The colors will be mixed
        :param max_value: Maximum value for the color ramp, if None, we consider max(grey)
        :param min_value: Minimum value for the color ramp, if None, we consider min(grey)
        :param grey_array: greyscale vector/matrix
        :param color1: Color for the minimum value
        :param color2: Color for the mid value
        :param color3: Color for the maximum value
        :param limits: Final boundary limits for the RGB values
        :return: Colored vector/matrix
        """

        # Get the color map by name:
        cm = plt.get_cmap(colormap)

        # normaliza dentro de min e max values
        grey_vector = (grey_array - min_value) / (max_value - min_value)

        # cut the values outside the limits of 0 and 1
        grey_vector[grey_vector < 0] = 0
        grey_vector[grey_vector > 1] = 1

        # Apply the colormap like a function to any array:
        colored_image = cm(grey_vector)

        return MinMaxScaler(limits).fit_transform(colored_image[:, 0:3])

        # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
        # But we want to convert to RGB in uint8 and save it:

        # original_shape = grey_array.shape
        #
        # grey_vector = grey_array.reshape(-1, 1)
        #
        # # normaliza dentro de min e max values
        # grey_vector = (grey_vector - min_value) / (max_value - min_value)
        #
        # # cut the values outside the limits of 0 and 1
        # grey_vector[grey_vector < 0] = 0
        # grey_vector[grey_vector > 1] = 1
        #
        # # invert the values because the HSV scale is inverted
        # grey_vector = grey_vector * (-1) + 1
        #
        # # limit the color to blue (if above 0.6 it goes to purple)
        # grey_vector = grey_vector * 0.6
        #
        # # grey2 = MinMaxScaler((0, 1)).fit_transform(grey)*(-1)+1
        # # grey2 = MinMaxScaler((0, 0.6)).fit_transform(grey2)
        #
        # ones = np.ones_like(grey_vector) * 0.8
        #
        # # create an hsv cube. the grayscale being the HUE
        # hsv = np.stack([grey_vector, ones, ones], axis=grey_vector.ndim)
        #
        # from skimage import color
        # rgb = color.hsv2rgb(hsv)
        # rgb = MinMaxScaler(limits).fit_transform(rgb.squeeze().reshape(-1,1))
        #
        # return rgb.reshape(-1,3)

        # select maximum and minimum values for the color ramp
        # max_value = max_value if max_value is not None else np.max(grey)
        # min_value = min_value if min_value is not None else np.min(grey)
        #
        # mid_point = np.mean(grey)
        #
        # # calculate the mixture in each pixel
        # # mixture 1 is for pixels below mid point
        # mixture1 = (grey-min_value)/(mid_point-min_value)
        #
        # # mixture 2 is for pixels above mid point
        # mixture2 = (grey-mid_point)/(max_value-mid_point)
        #
        # # get rid of mixtures above 1 and below 0
        # mixture1[mixture1 < 0] = 0
        # mixture1[mixture1 > 1] = 1
        #
        # mixture2[mixture2 < 0] = 0
        # mixture2[mixture2 > 1] = 1
        #
        # # add dimensions to the colors to match grey ndims+1 for correct broadcasting
        # color1 = np.array(color1)
        # color2 = np.array(color2)
        # for _ in range(grey.ndim):
        #     color1 = np.expand_dims(color1, axis=0)
        #     color2 = np.expand_dims(color2, axis=0)
        #     color3 = np.expand_dims(color3, axis=0)
        #
        # # add a last dimension to mixtures arrays
        # mixture1 = mixture1[..., np.newaxis]
        # mixture2 = mixture2[..., np.newaxis]
        #
        # # make the RGB color ramp between the 2 colors, based on the mixture
        # rgb_color_ramp = np.where(mixture1 < 1,
        #                           (1-mixture1)*color1 + mixture1*color2,
        #                           (1-mixture2)*color2 + mixture2*color3)
        #
        # scaled_rgb_color_ramp = MinMaxScaler(limits).fit_transform(rgb_color_ramp.reshape(-1, 1))
        #
        # return scaled_rgb_color_ramp.reshape(rgb_color_ramp.shape)

    @staticmethod
    def array2raster(filename, array, geo_transform, projection, nodatavalue=0, dtype=None):

        dtype = gdal.GDT_Float32 if dtype is None else dtype

        cols = array.shape[1]
        rows = array.shape[0]

        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(filename, cols, rows, 1, dtype, options=['COMPRESS=PACKBITS'])
        out_raster.SetGeoTransform(geo_transform)
        out_raster.SetProjection(projection)
        outband = out_raster.GetRasterBand(1)
        outband.SetNoDataValue(nodatavalue)
        outband.WriteArray(array)
        outband.FlushCache()
        print('Saving image: ' + filename)
        return

    @staticmethod
    def array2rgb_raster(filename, red, green, blue, geo_transform, projection, nodatavalue=-9999):

        cols = red.shape[1]
        rows = red.shape[0]

        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(filename, cols, rows, 3, gdal.GDT_Float32)
        out_raster.SetGeoTransform(geo_transform)
        out_raster.SetProjection(projection)
        outband = out_raster.GetRasterBand(1)
        outband.SetNoDataValue(nodatavalue)
        outband.WriteArray(red)

        outband = out_raster.GetRasterBand(2)
        outband.SetNoDataValue(nodatavalue)
        outband.WriteArray(green)

        outband = out_raster.GetRasterBand(3)
        outband.SetNoDataValue(nodatavalue)
        outband.WriteArray(blue)

        outband.FlushCache()
        print('Saving image: ' + filename)
        return

    @staticmethod
    def tif_2_pdf(tif_file, resolution=600, scale=2000):
        """Convert a TIF image into a PDF given a resolution"""

        pdf_file = tif_file[:-4] + '.pdf'

        print(f'Creating pdf file: {pdf_file}')
        translate = f'gdal_translate -outsize {resolution} 0 -ot Byte -scale 0 {scale} -of pdf ' \
                    + tif_file + ' ' + pdf_file
        os.system(translate)

        return pdf_file

    # -----------------------------------------------
    @staticmethod
    def array2multiband(filename, array, geo_transform, projection, nodatavalue=0, dtype=None):
        print('-----------------------------------------------')
        print('JE FAIS LE RASTER MULTILAYERS')

        dtype = gdal.GDT_Float32 if dtype is None else dtype

        cols = array[0].shape[1]
        rows = array[0].shape[0]
        print(cols, rows)
        nb_bands = len(array)

        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(filename, cols, rows, nb_bands, dtype, options=['COMPRESS=LZW'])
        out_raster.SetGeoTransform(geo_transform)
        out_raster.SetProjection(projection)

        # [out_raster.GetRasterBand(i) for i in range(0,nb_bands)]

        for i in range(0, nb_bands):
            outband = out_raster.GetRasterBand(i + 1)
            outband.SetNoDataValue(nodatavalue)
            outband.WriteArray(array[i])

        # outband = out_raster.GetRasterBand(2)
        # outband.SetNoDataValue(nodatavalue)
        # outband.WriteArray(green)
        #
        # outband = out_raster.GetRasterBand(3)
        # outband.SetNoDataValue(nodatavalue)
        # outband.WriteArray(blue)

        outband.FlushCache()
        print('Saving image: ' + filename)
        print('-----------------------------------------------')
        return

# ------------------------------------------------
    @staticmethod
    def get_train_test_data(data, train_size, min_train_size, max_train_size):
        """
        Split the provided data in train-test bunches
        :param min_train_size: minimum data quantity for train set
        :param max_train_size: maximum data quantity for train set
        :param train_size: percentage of the data to be used as train dataset
        :param data: data to be split
        :return: train and test datasets
        """
        dataset_size = data.shape[0]

        if (dataset_size * train_size) < min_train_size:
            train_size = min_train_size / dataset_size
            train_size = 1 if train_size > 1 else train_size

        elif (dataset_size * train_size) > max_train_size:
            train_size = max_train_size / dataset_size

        return train_test_split(data, train_size=train_size)

    @staticmethod
    def plot_clustered_data(data, cluster_names, file_name, graph_options, pdf_merger):
        plt.style.use('seaborn-whitegrid')

        plot_colors = ['goldenrod', 'darkorange', 'tomato', 'brown', 'gray', 'salmon', 'black', 'orchid', 'firebrick','orange', 'cyan']
        # plot_colors = list(colors.cnames.keys())

        fig, ax1 = plt.subplots()

        k = np.unique(data[:, 2])

        for i in k:
            cluster_i = data[data[:, 2] == i, 0:2]

            if int(i) in cluster_names.keys():
                label = cluster_names[int(i)]['name']
                colorname = cluster_names[int(i)]['color']
            else:
                label = 'Mixture'
                colorname = plot_colors[int(i)]

            ax1.set_xlabel(graph_options['x_label'])
            ax1.set_ylabel(graph_options['y_label'])
            ax1.set_title(graph_options['title'])

            ax1.plot(cluster_i[:, 0], cluster_i[:, 1], '.', label=label, c=colorname)

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)

        plt.savefig(file_name + '.png')

        if pdf_merger:
            plt.savefig(file_name + '.pdf')
            pdf_merger.append(file_name + '.pdf')

        # plt.show()
        plt.close()

        return

    @staticmethod
    def plot_graphs(bands, graphs_bands, labels_array, file_name, graph_title, invalid_mask=False, max_points=1000,
                    pdf_merger=None):

        # if combinations is not a list of lists, transform it in list of lists
        if type(graphs_bands[0]) == str:
            graphs_bands = [graphs_bands]

        for bands_names in graphs_bands:
            # O correto aqui e passar um dicionario com as opcoes, tipo, nome das legendas, etc.
            x_values = bands[bands_names[0]]
            y_values = bands[bands_names[1]]

            # create the graph filename
            graph_name = file_name + '_Graph_' + bands_names[0] + bands_names[1]

            # create the graph options dictionary
            graph_options = {'title': graph_title + ':' + bands_names[0] + 'x' + bands_names[1],
                             'x_label': bands_names[0],
                             'y_label': bands_names[1]}

            cluster_names = {1: {'name': 'Water', 'color': 'deepskyblue'},
                             2: {'name': 'Vegetation', 'color': 'forestgreen'}}

            # first, we will create the valid data array
            data = np.c_[x_values[~invalid_mask], y_values[~invalid_mask], labels_array[~invalid_mask]]

            plot_data, _ = DWutils.get_train_test_data(data, train_size=1, min_train_size=0, max_train_size=max_points)

            DWutils.plot_clustered_data(plot_data, cluster_names, graph_name, graph_options, pdf_merger)

        return

    @staticmethod
    def create_composite(bands, folder_name, pdf=True, resolution=600):

        # copy the RGB clipped bands to output directory

        red_band = copy(bands['Red'].GetDescription(), folder_name)
        green_band = copy(bands['Green'].GetDescription(), folder_name)
        blue_band = copy(bands['Blue'].GetDescription(), folder_name)

        composite_base_name = os.path.join(folder_name, os.path.split(folder_name)[-1] + '_composite')

        os.system('gdalbuildvrt -separate ' + composite_base_name + '.vrt ' +
                  red_band + ' ' + green_band + ' ' + blue_band)

        if pdf:
            cmd = f'gdal_translate -of pdf -ot Byte -scale 0 2000 -outsize {resolution} 0 ' \
                  + composite_base_name + '.vrt ' + composite_base_name + '.pdf'
            os.system(cmd)

        return composite_base_name

    @staticmethod
    def create_bands_dict(bands_array, bands_order):

        bands_dict = {}
        for i, band in enumerate(bands_order):
            bands_dict.update({band: bands_array[:,:,i]})

        return bands_dict

    @staticmethod
    def create_colorbar_pdf(product_name, title, label, colormap, min_value, max_value):
        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(4, 1))
        ax1 = fig.add_axes([0.05, 0.50, 0.90, 0.15])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.

        #norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        norm = matplotlib.colors.LogNorm(vmin=min_value, vmax=max_value)
        #norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=min_value, vmax=max_value)

        #
        # cdict = {'red': ((0.0, 0.0, 0.0),
        #                  (0.5, 0.0, 0.0),
        #                  (1.0, 1.0, 1.0)),
        #
        #          'green': ((0.0, 0.0, 0.0),
        #                    (0.5, 0.8, 0.8),
        #                    (1.0, 0.0, 0.0)),
        #
        #          'blue': ((0.0, 0.6, 1.0),
        #                   (0.5, 0.0, 0.0),
        #                   (1.0, 0.0, 0.0))}
        #
        # cmap = mpl.colors.LinearSegmentedColormap('custom', cdict)

        cmap = plt.get_cmap(colormap)

        # =========================

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                               norm=norm,
                                               orientation='horizontal')
        ax1.set_title(title)
        cb1.set_label('Legend: ' + label)

        plt.savefig(product_name)

    @staticmethod
    def find_file_glob(file_string, folder):

        file_list = [f for f in folder.iterdir() if file_string in f.stem]

        if len(file_list) > 0:
            return file_list[0]
        else:
            return None

    @staticmethod
    def read_gdal_ds(file, shape_file, temp_dir):
        """
        Read a GDAL dataset clipping it with a given shapefile, if necessary
        :param file: Filepath of the GDAL file (.tif, etc.) as Pathlib
        :param shape_file: file path of the shapefile
        :param temp_dir: file path of the temporary directory
        :return: GDAL dataset
        """
        gdal_mask = gdal.Open(file.as_posix())

        if gdal_mask and shape_file:

            opt = gdal.WarpOptions(cutlineDSName=shape_file, cropToCutline=True,
                                   srcNodata=-9999, dstNodata=-9999, outputType=gdal.GDT_Int16)

            dest_name = (temp_dir / (file.stem + '_clipped')).as_posix()
            clipped_mask_ds = gdal.Warp(destNameOrDestDS=dest_name,
                                        srcDSOrSrcDSTab=gdal_mask,
                                        options=opt)
            clipped_mask_ds.FlushCache()
            gdal_mask = clipped_mask_ds

        return gdal_mask

    @staticmethod
    def extract_angles_from_xml(xml):
        """
        Function to extract Zenith and Azimuth angles values from an xml Sentinel 2 file

        Parameters
        ----------
        xml : TYPE xml file
            DESCRIPTION Filepath of the metadata file from L2A Sentinel 2 data: example "SENTINEL2A_20200328-104846-345_L2A_T31TFJ_C_V2-2_MTD_ALL.xml"

        :return g: list of glint values

        Info
        -------
        SZA : TYPE float
            DESCRIPTION. Sun zenith angle
        SazA : TYPE float
            DESCRIPTION. Sun azimuth angle
        zenith_angle : TYPE list of strings
            DESCRIPTION. Mean_Viewing_Incidence_Angle_List for all the bands
        azimuth_angle : TYPE list of strings
            DESCRIPTION. Mean_Viewing_Incidence_Angle_List for all the bands

        """

        # Parsing xml file
        parser = etree.XMLParser()
        tree = etree.parse(xml, parser)

        root = tree.getroot()  # to iterate over the tree

        zenith_angle = root.xpath('//ZENITH_ANGLE[@unit="deg"]/text()')
        # zenith_angle[0] = sun_angle Mean_Viewing_Incidence_Angle_List for B2, B3....
        azimuth_angle = root.xpath('//AZIMUTH_ANGLE[@unit="deg"]/text()')
        # azimuth_angle[0] = sun_angle and then Mean_Viewing_Incidence_Angle_List for B2, B3....

        # the first values in the list correspond to the sun angle
        SZA = np.deg2rad(float(zenith_angle[0])) #radian
        SazA = float(azimuth_angle[0])

        # Remove sun angles (first element) from both lists
        zenith_angle.pop(0)
        azimuth_angle.pop(0)

        #return SZA, SazA, zenith_angle, azimuth_angle

        g = []

        for i in range(len(zenith_angle)):
            # Degrees to radian conversion
            viewA = np.deg2rad(float(zenith_angle[i]))
            phi = np.deg2rad((SazA - float(azimuth_angle[i])))

            Tetag = np.cos(viewA) * np.cos(SZA) - np.sin(viewA) * np.sin(SZA) * np.cos(phi)
            # Convert result to degrees
            g.append(np.degrees(np.arccos(Tetag)))

        return g

    @staticmethod
    def write_pdf(filename, text, size=(300, 50), position=(5, 5), font_color=(0, 0, 0)):
        out = Image.new("RGB", size, (255, 255, 255))
        draw = ImageDraw.Draw(out)

        draw.multiline_text(position, text, fill=font_color, anchor=None, spacing=10, align="center")

        out.save(filename)
        out.close()

        return filename

    @staticmethod
    def create_glint_pdf(xml, name_img, output_folder, g, pdf_merger):
        """
        Function to create an image to add in the pdf report that indicates if there is glint on an image

        Parameters
        ----------
        xml : TYPE xml file
            DESCRIPTION Filepath of the metadata file from L2A Sentinel 2 data: example "SENTINEL2A_20200328-104846-345_L2A_T31TFJ_C_V2-2_MTD_ALL.xml"
        current_imagename : getting current image name
        output_folder: filepath of the output folder
        g: TYPE list
            DESCRIPTION list with glint values for each band of the Sentinel 2 product
        pdf_merger: function to add an element to a pdf

        """
        # create an image
        out = Image.new("RGB", (300, 50), (255, 255, 255))
        # get a drawing context
        d = ImageDraw.Draw(out)
        # font size
        font = ImageFont.truetype("DejaVuSans.ttf", 16)

        # Test about glint values
        print("---------------------------")
        print("VALUES ANGLE GLINT")
        print(g)
        if min(g) < 20:
            print('GLINT SUR IMAGE ' + xml)
            # draw multiline text
            d.multiline_text((5, 5), "GLINT image \n" + name_img, fill=(0, 0, 0), font=font, anchor=None, spacing=0,
                             align="center")
        elif min(g) >= 20 and min(g) < 29:
            # values that may change
            print('MIGHT BE GLINT SUR IMAGE ' + xml)
            # draw multiline text
            d.multiline_text((5, 5), "MIGHT BE GLINT image \n" + name_img, fill=(0, 0, 0), font=font, anchor=None,
                             spacing=0, align="center")
        else:
            print("PAS DE GLINT SUR IMAGE " + xml)
            # draw multiline text
            d.multiline_text((5, 5), "NO GLINT image \n" + name_img, fill=(0, 0, 0), font=font, anchor=None,
                             spacing=0, align="center")
        print("---------------------------")

        # Printing details to obtain it in the log file when run on the cluster

        # name of the pdf image
        nameimg = "Glint_" + name_img
        # output filename
        filename = os.path.join(output_folder, nameimg)
        # Save as pdf
        out.save(filename + '.pdf')
        out.close()

        if pdf_merger:
            # Add to the main pdf
            pdf_merger.append(filename + '.pdf')

    @staticmethod
    def remove_negatives(bands, mask=None, negative_values='mask'):
        """
        Remove negatives values of given arrays b1 and b2, except masked values.

        :param bands: list of bands to be adjusted
        :param mask: initial mask
        :param negative_values: mask - mask the negative values; # fixed - replace all negative values for 0.001;
                shift - shift each band by its minimum value, so every band has only positive values;
                shift_all - shift each band by the minimum value of all bands. All bands will be shifted up by the
                same amount
        :return: nd arrays without negatives values
        """

        # If bands is not a list, create e list from it
        bands_list = bands if isinstance(bands, list) else [bands]

        # Create an empty list for the results
        results_list = []

        if negative_values == 'mask':
            for band in bands_list:
                # update the given mask
                mask[np.where(bands <= 0)] = 1
                results_list.append(np.where(band <= 0, -9999, band))

        elif negative_values == 'shift':
            for band in bands_list:
                # get the minimum value outside the mask (otherwise we could get -9999 for example)
                min_cte = np.min(band[~mask])
                min_cte = min_cte if min_cte < 0 else 0
                results_list.append(bands - min_cte)

        elif negative_values == 'shift_all':
            min_cte = np.min([np.min(band) for band in bands_list])
            min_cte = min_cte if min_cte < 0 else 0

            results_list = [band - min_cte for band in bands_list]

        else:
            print(f'Warning: negative values method {negative_values} not supported. Assuming fixed method')
            for band in bands_list:
                results_list.append(np.where(band <= 0, 0.001, band))

        # if there is just one band, return just the array, and not a list
        return results_list if len(results_list) > 1 else results_list[0]


