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


class DWutils:
    @staticmethod
    def check_path(path_str, is_dir=False):
        """
        Check if the path/file exists and returns a Path variable with it
        :param path_str: path string to test
        :param is_dir: whether if it is a directory or a file
        :return: Path type variable
        """
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
        outRaster = driver.Create(filename, cols, rows, 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform(geo_transform)
        outRaster.SetProjection(projection)
        outband = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(nodatavalue)
        outband.WriteArray(array)
        outband.FlushCache()
        print('Saving image: ' + filename)
        return

    @staticmethod
    def get_train_test_data(data, train_size, min_train_size, max_train_size):
        """
        Split the provided data in train-test bunches
        :param data: data to be splited
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

            ax1.plot(cluster_i[:, 0], cluster_i[:, 1], '.', label=label, c= colorname)

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)

        plt.savefig(file_name + '.png')

        #plt.show()
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
    dicS2BandNames = {'Blue': 'B2', 'Green': 'B3', 'Red': 'B4', 'Mir': 'B11', 'Mir2': 'B12', 'Nir': 'B8', 'Nir2': 'B8A'}
    dicL8USGSBandNames = {'Green': 'B3', 'Red': 'B4', 'Mir': 'B6', 'Nir': 'B5'}
    dicOtherBandNames = {'Green': 'band3', 'Red': 'band4', 'Mir': 'band6', 'Nir': 'band5'}

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

        return

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):

        if self._index == len(self.images)-1:
            raise StopIteration

        self._index += 1
        return self

    def current_image(self):

        return self.images[self._index]

    def name(self):

        return self.current_image().stem

    def find_product_bands(self):

        print('Retrieving bands for image: ' + self.current_image().as_posix())
        if self.product == 'S2_THEIA':
            # get flat reflectance bands in a list
            bands = [file for file in self.current_image().iterdir() if
                     file .suffix == '.tif' and 'FRE' in file.stem]
            for b in bands:
                print(b.stem)
        else:
            bands = None

        return bands

    def open_image(self, ref_band_name='Red'):
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
            else:
                band_names = self.dicOtherBandNames
        return band_names

    def get_projection(self):
        return self._ref_band.GetProjection()

    def get_geo_transform(self):
        return self._ref_band.GetGeoTransform()

    def get_x_size(self):
        return self._ref_band.RasterXSize

    def get_y_size(self):
        return self._ref_band.RasterYSize

    @staticmethod
    def open_gdal_image(bands_list, desired_band):
        """
        Get the image in the list corresponding to the informed Band.
        Return the image opened with GDAL as a RasterImage object
        If cant find the band return None
        If is more than 1 image, raise exception
        """
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

        return

    def load_raster_bands(self, bands_list: list):

        if len(self.gdal_bands) == 0:
            raise OSError('Dataset not opened or no bands available')

        x_size, y_size = self.get_x_size(), self.get_y_size()

        for band in bands_list:

            if band not in self.raster_bands and band in self.gdal_bands:

                gdal_img = self.gdal_bands[band]

                raster_band = gdal_img.ReadAsArray(buf_xsize=x_size, buf_ysize=y_size).astype(dtype=np.float32) / 10000
                self.raster_bands.update({band: raster_band})

                self.invalid_mask |= raster_band == -9999

        return self.raster_bands

    def update_mask(self, mask):
        self.invalid_mask |= mask

        return self.invalid_mask


class DWSaver:
    def __init__(self, output_folder, image_name, product_name, geo_transform, projection, area_name=None):

        self.output_folder = self.create_output_folder(output_folder, image_name, area_name)
        self.base_name = product_name + '-' + image_name.split('_')[1]

        self.geo_transform = geo_transform
        self.projection = projection

        self._temp_dir = None

        return

    @staticmethod
    def create_output_folder(output_folder, image_name, area_name):
        if not area_name:
            output_folder = output_folder.joinpath(image_name)
        else:
            output_folder = output_folder.joinpath(area_name).joinpath(image_name)

        output_folder.mkdir(exist_ok=True)

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
            self._temp_dir = self.output_folder/'temp_dir'
            self._temp_dir.mkdir(exist_ok=True)

        return self._temp_dir