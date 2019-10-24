import os
import sys
import numpy as np
from osgeo import gdal
from pathlib import Path
from DWCommon import DWutils


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

