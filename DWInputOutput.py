import os
import sys
import numpy as np
from osgeo import gdal
from pathlib import Path
from sklearn.model_selection import train_test_split
from shutil import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import configparser
import ast


class DWConfig:

    def __init__(self, config_file=None):

        self.config = self.load_config_file(config_file)

        return

    @staticmethod
    def load_config_file(config_file=None):

        if not config_file:
            config_file = 'WaterDetect.ini'

        print('Loading configuration file {}'.format(config_file))
        DWutils.check_path(config_file)

        config = configparser.ConfigParser()
        config.read('WaterDetect.ini')

        return config

    @property
    def reference_band(self):
        return self.config.get('General', 'reference_band')

    @property
    def create_composite(self):
        return ast.literal_eval(self.config.get('General', 'create_composite'))

    @property
    def clustering_method(self):
        return self.config.get('Clustering', 'clustering_method')

    @property
    def train_size(self):
        return ast.literal_eval(self.config.get('Clustering', 'train_size'))

    @property
    def min_train_size(self):
        return ast.literal_eval(self.config.get('Clustering', 'min_train_size'))

    @property
    def max_train_size(self):
        return ast.literal_eval(self.config.get('Clustering', 'max_train_size'))

    @property
    def clip_band(self):
        band = self.config.get('Clustering', 'clip_band')

        if band == 'None' or band == 'none' or band == '':
            return None
        else:
            return band

    @property
    def clip_value(self):
        return ast.literal_eval(self.config.get('Clustering', 'clip_value'))

    @property
    def score_index(self):
        return self.config.get('Clustering', 'score_index')

    @property
    def classifier(self):
        return self.config.get('Clustering', 'classifier')

    @property
    def detect_water_cluster(self):
        return self.config.get('Clustering', 'detectwatercluster')

    @property
    def min_clusters(self):
        return ast.literal_eval(self.config.get('Clustering', 'min_clusters'))

    @property
    def max_clusters(self):
        return ast.literal_eval(self.config.get('Clustering', 'max_clusters'))

    @property
    def graphs_bands(self):

        bands_str = self.config.get('Graphs', 'graphs_bands')

        bands_lst = ast.literal_eval(bands_str)

        # if bands_keys is not a list of lists, transform it
        if type(bands_lst[0]) == str:
            bands_lst = [bands_lst]

        return bands_lst

    @property
    def clustering_bands(self):

        bands_str = self.config.get('Clustering', 'clustering_bands')

        bands_lst = ast.literal_eval(bands_str)

        # if bands_keys is not a list of lists, transform it
        if type(bands_lst[0]) == str:
            bands_lst = [bands_lst]

        return bands_lst


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
    def calc_normalized_difference(img1, img2):
        nd = (img1-img2) / (img1 + img2)

        nd[nd > 1] = 1
        nd[nd < -1] = -1

        nd_mask = np.isinf(nd) | np.isnan(nd)
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
    def plot_clustered_data(data, cluster_names, file_name, graph_options):
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

        # plt.show()
        plt.close()

        return

    @staticmethod
    def plot_graphs(bands, bands_combination, labels_array, file_name, invalid_mask=False, max_points=1000):

        # if combinations is not a list of lists, transform it in list of lists
        if type(bands_combination[0]) == str:
            bands_combination = [bands_combination]

        for bands_names in bands_combination:
            # O correto aqui e passar um dicionario com as opcoes, tipo, nome das legendas, etc.
            x_values = bands[bands_names[0]]
            y_values = bands[bands_names[1]]

            # create the graph filename
            graph_name = file_name + '_Graph_' + bands_names[0] + bands_names[1]

            # create the graph options dictionary
            graph_options = {'title': 'Scatterplot ' + bands_names[0] + ' x ' + bands_names[1],
                             'x_label': bands_names[0],
                             'y_label': bands_names[1]}

            cluster_names = {1: {'name': 'Water', 'color': 'deepskyblue'},
                             2: {'name': 'Vegetation', 'color': 'forestgreen'}}

            # first, we will create the valid data array
            data = np.c_[x_values[~invalid_mask], y_values[~invalid_mask], labels_array[~invalid_mask]]

            plot_data, _ = DWutils.get_train_test_data(data, train_size=1, min_train_size=0, max_train_size=max_points)

            DWutils.plot_clustered_data(plot_data, cluster_names, graph_name, graph_options)

        return

    @staticmethod
    def create_composite(bands, folder_name):

        # copy the RGB clipped bands to output directory

        redband = copy(bands['Red'].GetDescription(), folder_name)
        greenband = copy(bands['Green'].GetDescription(), folder_name)
        blueband = copy(bands['Blue'].GetDescription(), folder_name)

        compositename = os.path.join(folder_name, os.path.split(folder_name)[-1] + '_composite.vrt')

        os.system('gdalbuildvrt -separate ' + compositename + ' ' +
                  redband + ' ' + greenband + ' ' + blueband)

        return


class DWLoader:
    dicS2BandNames = {'Blue': 'B2', 'Green': 'B3', 'Red': 'B4', 'Mir': 'B11', 'Mir2': 'B12',
                      'Nir': 'B8', 'Nir2': 'B8A'}

    dicSen2CorBandNames = {'Blue': 'B02', 'Green': 'B03', 'Red': 'B04', 'Mir': 'B11', 'Mir2': 'B12',
                           'Nir': 'B08', 'Nir2': 'B8A'}

    dicL8USGSBandNames = {'Green': 'B3', 'Red': 'B4', 'Mir': 'B6', 'Nir': 'B5'}

    dicOtherBandNames = {'Blue': 'band2', 'Green': 'band3', 'Red': 'band4',
                         'Mir': 'band6', 'Nir': 'band5', 'Mir2': 'band7'}

    def __init__(self, input_folder, shape_file, product):

        self.input_folder = DWutils.check_path(input_folder, is_dir=True)
        self.shape_file = DWutils.check_path(shape_file, is_dir=False)

        self.images = DWutils.get_directories(self.input_folder)

        self.product = product

        # index for iterating through DWLoader
        self._index = 0

        # dictionary of bands pointing to gdal images
        self.gdal_bands = None
        self._clipped_gdal_bands = None

        # dictionary of bands pointing to in memory arrays
        self.raster_bands = None

        # reference band for shape, projection and transformation (the first band loaded by the user)
        self._ref_band = None

        # mask with the invalid pixels
        self.invalid_mask = False

        # temporary directory for clipping bands
        self.temp_dir = None

        return

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):

        if self._index == len(self.images)-1:
            raise StopIteration

        self._index += 1
        return self

    @property
    def area_name(self):
        """
        Extracts the name of the area based on the shapefile name
        :return: name of the area
        """

        if self.shape_file:
            return self.shape_file.stem
        else:
            return None

    def current_image(self):

        return self.images[self._index]

    @property
    def name(self):

        return self.current_image().stem

    def find_product_bands(self):

        print('Retrieving bands for image: ' + self.current_image().as_posix())
        if self.product == 'S2_THEIA':
            # get flat reflectance bands in a list
            bands = [file for file in self.current_image().iterdir() if
                     file .suffix == '.tif' and 'FRE' in file.stem]

        elif self.product == 'LANDSAT8':
            bands = [file for file in self.current_image().iterdir() if
                     file .suffix == '.tif' and 'sr_band' in file.stem]

        elif self.product == 'SEN2COR':
            bands = [file for file in self.current_image().iterdir() if
                     file .suffix == '.jp2' and ('_20m' in file.stem or '_10m' in file.stem)]
        else:
            bands = None

        if bands:
            for b in bands:
                print(b.stem)

        return bands

    def open_current_image(self, ref_band_name='Red'):
        """
        Load a bands list, given a image_list and a dictionary of Keys(BandName) and identifiers to parse the filename
        ex. {'Green':'B3', 'Red':'B4'...}
        The result, will be a dictionary with Keys(BandName) and RasterImages as values
        """
        # reset gdal and raster bands
        self.gdal_bands = {}
        self._clipped_gdal_bands = {}
        self.raster_bands = {}
        self.invalid_mask = False

        print('Opening image in loader')
        product_bands = self.find_product_bands()
        product_bands_names = self.get_bands_names()

        self.gdal_bands = {}
        for band_name in product_bands_names:
            print('Loading band: ' + band_name)
            gdal_img = self.open_gdal_image(product_bands, product_bands_names[band_name])
            self.gdal_bands.update({band_name: gdal_img})

        # assign the reference band to _ref_band
        self._ref_band = self.gdal_bands[ref_band_name]

        return self.gdal_bands

    def get_bands_names(self):
        if self.product in ["L8_THEIA", "S5_THEIA"]:
            print('not yet implemented')
            sys.exit()

        else:
            if 'L8_USGS' in self.product:
                band_names = self.dicL8USGSBandNames
            elif self.product in ["S2_PEPS", "S2_S2COR", "S2_THEIA", "S2_L2H"]:
                band_names = self.dicS2BandNames
            elif self.product in ["SEN2COR"]:
                band_names = self.dicSen2CorBandNames
            else:
                band_names = self.dicOtherBandNames
        return band_names

    @property
    def projection(self):
        return self._ref_band.GetProjection()

    @property
    def geo_transform(self):
        return self._ref_band.GetGeoTransform()

    @property
    def x_size(self):
        return self._ref_band.RasterXSize

    @property
    def y_size(self):
        return self._ref_band.RasterYSize

    @staticmethod
    def open_gdal_image(bands_list, desired_band):
        """
        Get the image in the list corresponding to the informed Band.
        Return the image opened with GDAL as a RasterImage object
        If cant find the band return None
        If is more than 1 image, raise exception
        """
        # todo: the desired band depends on the product
        desired_band = '_' + desired_band + '.'
        image_band = list(filter(lambda x: desired_band in os.path.split(x)[-1], bands_list))

        if len(image_band) == 0:
            return None, None
        elif len(image_band) > 1:
            raise OSError('More than one band {} in image list'.format(desired_band))

        gdal_ds = gdal.Open(image_band[0].as_posix())

        if not gdal_ds:
            raise OSError("Couldn't open band file {}".format(image_band[0]))

        return gdal_ds

    def clip_bands(self, bands_to_clip, ref_band, temp_dir):

        opt = gdal.WarpOptions(cutlineDSName=self.shape_file, cropToCutline=True,
                               srcNodata=-9999, dstNodata=-9999, outputType=gdal.GDT_Float32)

        for band in bands_to_clip:
            if band not in self._clipped_gdal_bands and band in self.gdal_bands:

                dest_name = (temp_dir/(band+'.tif')).as_posix()
                self._clipped_gdal_bands.update({band: gdal.Warp(destNameOrDestDS=dest_name,
                                                                 srcDSOrSrcDSTab=self.gdal_bands[band],
                                                                 options=opt)})
                self.gdal_bands[band] = self._clipped_gdal_bands[band]
                self.gdal_bands[band].FlushCache()

        self._ref_band = self.gdal_bands[ref_band]
        self.temp_dir = temp_dir

        return

    def load_raster_bands(self, bands_list: list):

        if len(self.gdal_bands) == 0:
            raise OSError('Dataset not opened or no bands available')

        for band in bands_list:

            if band not in self.raster_bands and band in self.gdal_bands:

                gdal_img = self.gdal_bands[band]

                raster_band = gdal_img.ReadAsArray(buf_xsize=self.x_size,
                                                   buf_ysize=self.y_size).astype(dtype=np.float32) / 10000
                self.raster_bands.update({band: raster_band})

                self.invalid_mask |= raster_band == -0.9999

        return self.raster_bands

    def update_mask(self, mask):
        self.invalid_mask |= mask

        return self.invalid_mask

    def load_masks(self):

        mask_processor = None
        if self.product == 'S2_THEIA':
            mask_processor = DWTheiaMaskProcessor(self.current_image(), self.x_size, self.y_size,
                                                  self.shape_file, self.temp_dir)
        elif self.product == 'LANDSAT8':
            mask_processor = DWTheiaMaskProcessor(self.current_image(), self.x_size, self.y_size,
                                                  self.shape_file, self.temp_dir)

        if mask_processor:
            self.update_mask(mask_processor.get_combined_masks())

        # if self.product == 'S2_THEIA':
        #     mask_folder = self.current_image()/'MASKS'
        #     cloud_mask_file = [file for file in mask_folder.glob('*_CLM_R1.tif')][0]
        #
        #     cloud_mask_ds = gdal.Open(cloud_mask_file.as_posix())
        #
        #     # todo: make the clipping function generic to work with masks
        #
        #     # if there are clipped bands, we have to clip the masks as well
        #     if self._clipped_gdal_bands:
        #         opt = gdal.WarpOptions(cutlineDSName=self.shape_file, cropToCutline=True,
        #                                srcNodata=-9999, dstNodata=-9999, outputType=gdal.GDT_Int16)
        #
        #         dest_name = (self.temp_dir/'CLM_R1.tif').as_posix()
        #         cloud_mask_ds = gdal.Warp(destNameOrDestDS=dest_name,
        #                                   srcDSOrSrcDSTab=cloud_mask_ds,
        #                                   options=opt)
        #         cloud_mask_ds.FlushCache()
        #
        #     cloud_mask = cloud_mask_ds.ReadAsArray(buf_xsize=self.x_size, buf_ysize=self.y_size)
        #
        #     new_mask |= (cloud_mask == -9999)
        #     new_mask |= (np.bitwise_and(cloud_mask, theia_cloud_mask['all_clouds_and_shadows']) != 0)
        #

        return self.invalid_mask


class DWSaver:
    def __init__(self, output_folder, product_name, area_name=None):

        # save the base output folder (root of all outputs)
        self.base_output_folder = DWutils.check_path(output_folder, is_dir=True)

        # save the product name
        self.product_name = product_name

        # save the name of the area
        self.area_name = area_name

        # initialize other objects variables
        self._temp_dir = None
        self.base_name = None
        self.geo_transform = None
        self.projection = None
        self.output_folder = None

        return

    def set_output_image(self, image_name, geo_transform, projection):
        """
        For each image, the saver has to prepare the specific output directory, and saving parameters.
        The output directory is based on the base_output_folder, the area name and the image name
        :param image_name: name of the image being processed
        :param geo_transform: geo transformation to save rasters
        :param projection: projection to save rasters
        :return: Nothing
        """

        self.output_folder = self.create_output_folder(self.base_output_folder,
                                                       image_name,
                                                       self.area_name)

        self.base_name = self.create_base_name(self.product_name, image_name)

        self.geo_transform = geo_transform
        self.projection = projection

        self.base_name = self.create_base_name(self.product_name, image_name)

        return

    def update_geo_transform(self, geo_transform, projection):

        self.geo_transform = geo_transform
        self.projection = projection

        return

    @staticmethod
    def create_base_name(product_name, image_name):
        return product_name + '-' + image_name.split('_')[1]

    @staticmethod
    def create_output_folder(output_folder, image_name, area_name):
        if not area_name:
            output_folder = output_folder.joinpath(image_name)
        else:
            output_folder = output_folder.joinpath(area_name).joinpath(image_name)

        output_folder.mkdir(parents=True, exist_ok=True)

        return output_folder

    def save_array(self, array, name, opt_relative_path=None):

        if opt_relative_path:
            filename = self.output_folder.joinpath(opt_relative_path)
            filename.mkdir(exist_ok=True)
        else:
            filename = self.output_folder

        filename = filename.joinpath(name + '.tif').as_posix()

        DWutils.array2raster(filename, array, self.geo_transform, self.projection)

    @property
    def temp_dir(self):

        if not self._temp_dir:
            self._temp_dir = self.output_folder / 'temp_dir'
            self._temp_dir.mkdir(exist_ok=True)

        return self._temp_dir


class DWTheiaMaskProcessor:

    TheiaMaskDict = {'CLM': '*_CLM_R2.tif',
                     'EDG': '*_EDG_R2.tif',
                     'MG2': '*_MG2_R2.tif',
                     'SAT1': '*_SAT_R1.tif',
                     'SAT2': '*_SAT_R2.tif'}

    TheiaCLMDict = {'all_clouds_and_shadows': 1 << 0,
                    'all_clouds': 1 << 1,
                    'clouds_blue_band': 1 << 2,
                    'clouds_multi_temporal': 1 << 3,
                    'cirrus': 1 << 4,
                    'cloud_shadows': 1 << 5,
                    'other_shadows': 1 << 6,
                    'high_clouds': 1 << 7}

    TheiaMG2Dict = {'water': 1 << 0,
                    'all_clouds': 1 << 1,
                    'snow': 1 << 2,
                    'cloud_shadows': 1 << 3,
                    'other_shadows': 1 << 4,
                    'terrain_mask': 1 << 5,
                    'sun_too_low': 1 << 6,
                    'sun_tangent': 1 << 7}

    def __init__(self, base_folder, x_size, y_size, shape_file=None, temp_dir=None):

        self.x_size = x_size
        self.y_size = y_size

        self.masks_folder = Path(base_folder)/'MASKS'

        self.masks = self.open_masks(shape_file, temp_dir)


        return

    def open_masks(self, shape_file, temp_dir):

        masks = {}
        gdal_masks = self.open_gdal_masks(shape_file, temp_dir)

        for mask_key, mask_ds in gdal_masks.items():
            masks.update({mask_key: mask_ds.ReadAsArray(buf_xsize=self.x_size, buf_ysize=self.y_size)})

        return masks

    def open_gdal_masks(self, shape_file, temp_dir):

        gdal_masks = {}

        for mask_key, mask_name in self.TheiaMaskDict.items():
            mask_file = [file for file in self.masks_folder.glob(mask_name)][0]
            gdal_masks.update({mask_key: gdal.Open(mask_file.as_posix())})

        if shape_file:

            opt = gdal.WarpOptions(cutlineDSName=shape_file, cropToCutline=True,
                                   srcNodata=-9999, dstNodata=-9999, outputType=gdal.GDT_Int16)

            for mask_key, mask_ds in gdal_masks.items():
                dest_name = (temp_dir / mask_key).as_posix()
                clipped_mask_ds = gdal.Warp(destNameOrDestDS=dest_name,
                                            srcDSOrSrcDSTab=mask_ds,
                                            options=opt)
                clipped_mask_ds.FlushCache()
                gdal_masks.update({mask_key: clipped_mask_ds})

        return gdal_masks

    def get_combined_masks(self):

        # cloud_mask = np.bitwise_and(self.masks['CLM'], theia_cloud_mask['all_clouds_and_shadows']) != 0
        cloud_mask = np.bitwise_and(self.masks['CLM'], self.TheiaCLMDict['all_clouds_and_shadows']) != 0

        # take care of -9999 ?
        edg_mask = self.masks['EDG'] != 0

        # SAT
        sat_mask = (self.masks['SAT1'] != 0) | (self.masks['SAT2'] != 0)

        # MG2 masks out snow and other shadows
        mg2_mask = np.bitwise_and(self.masks['MG2'],
                                  self.TheiaMG2Dict['snow'] |
                                  self.TheiaMG2Dict['other_shadows'] |
                                  self.TheiaMG2Dict['terrain_mask']) != 0

        # return cloud_mask | edg_mask | sat_mask | mg2_mask
        return edg_mask

# considering a bit mask like:
# mask = 0_0_0_1_0_1_0_0   (bit2 = cloud, bit 4 = shadow)
# if we want the pixels clear from any of the mask bits, we should perform bitwise_and == 0
# pixel= 0_0_0_0_0_0_0_0 -> and = 0_0_0_0_0_0_0_0 (== 0 -> clear)
# pixel= 0_1_0_1_0_1_0_1 -> and = 0_0_0_1_0_1_0_0 (!= 0 -> masked out)
# pixel= 0_0_0_1_0_0_1_1 -> and = 0_0_0_1_0_0_0_0 (!= 0 -> masked out)

# if we want the pixels clear from both mask bits at the same time, we do the same but
# bitwise_and != MASK
# mask = 0_0_0_1_0_1_0_0   (bit2 = cloud, bit 4 = shadow)
# pixel= 0_1_0_1_0_1_0_1 -> and = 0_0_0_1_0_1_0_0 (== mask -> masked out)
# pixel= 0_0_0_1_0_0_1_1 -> and = 0_0_0_1_0_0_0_0 (!= mask -> clear)

# if we want the pixels masked by any of the mask bits we do bitwise_and != 0
# mask = 0_0_0_1_0_1_0_0   (bit2 = cloud, bit 4 = shadow)
# pixel= 0_0_0_0_0_0_0_0 -> and = 0_0_0_0_0_0_0_0 (== 0 -> not masked)
# pixel= 0_1_0_1_0_1_0_1 -> and = 0_0_0_1_0_1_0_0 (!= 0 -> masked out)
# pixel= 0_0_0_1_0_0_1_1 -> and = 0_0_0_1_0_0_0_0 (!= 0 -> masked out)

