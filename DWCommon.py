import os
from shutil import copy
import configparser
import ast
from sklearn.model_selection import train_test_split
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
        nd = (img1-img2) / (img1 + img2)

        nd[nd > 1] = 1
        nd[nd < -1] = -1

        nd_mask = np.isinf(nd) | np.isnan(nd) | mask
        nd = np.ma.array(nd, mask=nd_mask, fill_value=-9999)

        return nd.filled(), nd.mask

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
