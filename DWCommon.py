import os
from shutil import copy
import configparser
import ast

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

from osgeo import gdal
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DWConfig:

    _config_file = 'WaterDetect.ini'
    _defaults = {'reference_band': 'Red',
                 'create_composite': 'True',
                 'pdf_reports': 'False',
                 'texture_streching': 'False',
                 'maximum_invalid': '0.8',
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
                 'graphs_bands': "[['mbwi', 'mndwi'], ['ndwi', 'mbwi']]"
                 }

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
            return ast.literal_eval(str_value)
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
    def texture_stretching(self):
        return self.get_option('General', 'texture_stretching', evaluate=True)

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
    def clustering_method(self):
        return self.get_option('Clustering', 'clustering_method', evaluate=False)

    @property
    def train_size(self):
        return self.get_option('Clustering', 'train_size', evaluate=True)

    @property
    def min_train_size(self):
        return self.get_option('Clustering', 'min_train_size',evaluate=True)

    @property
    def max_train_size(self):
        return self.get_option('Clustering', 'max_train_size', evaluate=True)

    @property
    def clip_band(self):
        band = self.get_option('Clustering', 'clip_band', evaluate=False)

        if band == 'None' or band == 'none' or band == '':
            return None
        else:
            return band

    @property
    def clip_value(self):
        return self.get_option('Clustering', 'clip_value', evaluate=True)

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
    def clustering_bands(self):

        bands_lst = self.get_option('Clustering', 'clustering_bands', evaluate=True)

        # if bands_keys is not a list of lists, transform it
        if type(bands_lst[0]) == str:
            bands_lst = [bands_lst]

        return bands_lst

    def get_masks_list(self, product):

        masks_lst = []

        if product == 'LANDSAT8':
            section_name = 'LandsatMasks'
        else:
            section_name = 'TheiaMasks'

        for key in self.config._sections[section_name]:
            if self.config.getboolean(section_name, key):
                masks_lst.append(key)

        return masks_lst


class DWutils:

    @staticmethod
    def bitwise_or(array, bit_values):
        return np.bitwise_or(array, bit_values)

    @staticmethod
    def bitwise_and(array, bit_values):
        return np.bitwise_and(array, bit_values)

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
    def calc_normalized_difference(img1, img2, mask=False):
        """
        Calc the normalized difference of given arrays (img1 - img2)/(img1 + img2).
        Updates the mask if any invalid numbers (ex. np.inf or np.nan) are encountered
        :param img1: first array
        :param img2: second array
        :param mask: initial mask, that will be updated
        :return: nd array filled with -9999 in the mask and the mask itself
        """

        # if any of the bands is set to zero in the pixel, makes a small shift upwards, as proposed by olivier hagole
        # https://github.com/olivierhagolle/modified_NDVI
        nd = np.where((img1 > 0) & (img2 > 0), (img1-img2) / (img1 + img2),
                      (img1+0.005-img2+0.005) / (img1+0.005 + img2+0.005))

        # nd = (img1-img2) / (img1 + img2)

        nd[nd > 1] = 1
        nd[nd < -1] = -1

        # if result is infinite, result should be 1
        nd[np.isinf(nd)] = 1

        # nd_mask = np.isinf(nd) | np.isnan(nd) | mask
        nd_mask = np.isnan(nd) | mask

        nd = np.ma.array(nd, mask=nd_mask, fill_value=-9999)

        return nd.filled(), nd.mask

    @staticmethod
    def rgb_burn_in(red, green, blue, burn_in_array, color=None, min_value=None, max_value=None, colormap='viridis',
                    fade=1, uniform_distribution=False, no_data_value=-9999):
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
        :return: RGB image bands
        """

        if color:
            new_red = np.where(burn_in_array == no_data_value, red*fade, color[0])
            new_green = np.where(burn_in_array == no_data_value, green*fade, color[1])
            new_blue = np.where(burn_in_array == no_data_value, blue*fade, color[2])

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
                                                         colormap=colormap, limits=(0, 0.14))

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
    def array2raster(filename, array, geo_transform, projection, nodatavalue=0):

        cols = array.shape[1]
        rows = array.shape[0]

        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
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

        plot_colors = ['goldenrod', 'darkorange', 'tomato', 'brown', 'gray', 'salmon', 'black', 'orchid', 'firebrick']
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
    def create_composite(bands, folder_name, pdf=True):

        # copy the RGB clipped bands to output directory

        red_band = copy(bands['Red'].GetDescription(), folder_name)
        green_band = copy(bands['Green'].GetDescription(), folder_name)
        blue_band = copy(bands['Blue'].GetDescription(), folder_name)

        composite_base_name = os.path.join(folder_name, os.path.split(folder_name)[-1] + '_composite')

        os.system('gdalbuildvrt -separate ' + composite_base_name + '.vrt ' +
                  red_band + ' ' + green_band + ' ' + blue_band)

        if pdf:
            cmd = 'gdal_translate -of pdf -ot Byte -scale 0 2000 -outsize 600 0 ' + composite_base_name + '.vrt ' \
                  + composite_base_name + '.pdf'
            os.system(cmd)

        return composite_base_name

    @staticmethod
    def create_bands_dict(bands_array, bands_order):

        bands_dict = {}
        for i, band in enumerate(bands_order):
            bands_dict.update({band: bands_array[:,:,i]})

        return bands_dict


    @staticmethod
    def create_colorbar_pdf(product_name, title, colormap, min_value, max_value):
        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(4, 1))
        ax1 = fig.add_axes([0.05, 0.50, 0.90, 0.15])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.

        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
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
        cb1.set_label(title + ' Legend')

        plt.savefig(product_name)