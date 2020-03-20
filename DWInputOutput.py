import os
import sys
import numpy as np
from osgeo import gdal
from pathlib import Path
from DWCommon import DWutils


class DWLoader:

    dicS2_THEIA = {'bands_names': {'Blue': 'B2', 'Green': 'B3', 'Red': 'B4', 'Mir': 'B11', 'Mir2': 'B12',
                   'Nir': 'B8', 'Nir2': 'B8A', 'RedEdg1': 'B5', 'RedEdg2': 'B6', 'RedEdg3': 'B7'},
                   'suffix': '.tif', 'string': 'SRE'}

    dicSEN2COR = {'bands_names': {'Blue': 'B02', 'Green': 'B03', 'Red': 'B04', 'Mir': 'B11', 'Mir2': 'B12'},
                  'Nir': 'B08', 'Nir2': 'B8A'}

    dicL8USGS = {'bands_names': {'Green': 'B3', 'Red': 'B4', 'Mir': 'B6', 'Nir': 'B5'}}

    dicLANDSAT = {'bands_names': {'Aero': 'band1', 'Blue': 'band2', 'Green': 'band3', 'Red': 'band4',
                  'Mir': 'band6', 'Nir': 'band5', 'Mir2': 'band7'},
                  'suffix': '.tif', 'string': 'sr_band'}

    dicS2_L1C = {'bands_names': {'Blue': 'B02', 'Green': 'B03', 'Red': 'B04', 'Mir': 'B11', 'Mir2': 'B12',
                                 'Nir': 'B08', 'Nir2': 'B8A'},
                 'suffix': '.jp2',
                 'string': '',
                 'subdir': 'GRANULE/*/IMG_DATA'}

    def __init__(self, input_folder, shape_file, product):

        # save the input folder (holds all the images) and the shapefile
        self.input_folder = DWutils.check_path(input_folder, is_dir=True)
        self.shape_file = DWutils.check_path(shape_file, is_dir=False)

        # load all sub-directories in the input folder (images) in a list
        self.images = DWutils.get_directories(self.input_folder)

        # the product indicates if the images are S2_THEIA, LANDSAT, SEN2COR, etc.
        self.product = product

        # index for iterating through the images list. Starts with the first image
        self._index = 0

        # dictionary of bands pointing to gdal images
        self.gdal_bands = None
        self._clipped_gdal_bands = None

        # dictionary of bands pointing to in memory numpy arrays
        self.raster_bands = None

        # reference band for shape, projection and transformation as a GDAL object
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
    def product_dict(self):

        if self.product in ["L8_THEIA", "S5_THEIA"]:
            print('Product not yet implemented')
            sys.exit()

        else:
            if 'L8_USGS' in self.product:
                product_dict = self.dicL8USGS
            elif self.product in ["S2_PEPS", "S2_S2COR", "S2_THEIA", "S2_L2H"]:
                product_dict = self.dicS2_THEIA
            elif self.product in ["SEN2COR"]:
                product_dict = self.dicSEN2COR
            elif self.product in ["S2_L1C"]:
                product_dict = self.dicS2_L1C
            else:
                product_dict = self.dicLANDSAT

        return product_dict

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

    @property
    def current_image_folder(self):
        """
        Returns the full path folder of current (selected) image
        :return: Posixpath of current image
        """
        return self.images[self._index]

    @property
    def current_image_name(self):
        """
        Returns the name of the current (selected) image
        :return: String name of the current (selected) image
        """
        return self.current_image_folder.stem

    @property
    def bands_path(self):
        """
        Return the directory of the bands depending on the product
        :return: PosixPath of the directory containing the bands
        """

        bands_path = self.current_image_folder

        # if there is a subdir for the product, append it to the path
        # caution with the use of '*' that represents an unnamed directory
        if 'subdir' in self.product_dict.keys():
            split_subdir = self.product_dict['subdir'].split('/')

            for subdir in split_subdir:
                if subdir  != '*':
                    bands_path /= subdir
                else:
                    # add the first directory (hopefully the only one) to the path
                    bands_path /= [d for d in bands_path.iterdir() if d.is_dir()][0]

        return bands_path

    def get_bands_files(self):
        """
        Retrieve the full path of bands saved for the current image, according to the product
        :return: Posixpath of bands files
        """
        print('Retrieving bands for image: ' + self.current_image_folder.as_posix())

        # put the full path of the corresponding bands in a list
        bands = [file for file in self.bands_path.iterdir() if file.suffix == self.product_dict['suffix']
                 and self.product_dict['string'] in file.stem]

        if bands:
            print('The following bands were found:')
            for b in bands:
                print(b.name)
        else:
            raise OSError('No bands found in dir: ')

        return bands

    def open_current_image(self, ref_band_name='Red'):
        """
        Load a bands list, given a image_list and a dictionary of Keys(BandName) and identifiers to parse the filename
        ex. {'Green':'B3', 'Red':'B4'...}
        The result, will be a dictionary with Keys(BandName) and GdalDatasets as values
        """
        # reset gdal and raster bands
        self.gdal_bands = {}
        self._clipped_gdal_bands = {}
        self.raster_bands = {}
        self.invalid_mask = False

        print('Opening image in loader')
        bands_files = self.get_bands_files()

        for band_name in self.product_dict['bands_names']:
            print('Loading band: ' + band_name)
            gdal_img = self.open_gdal_image(bands_files, self.product_dict['bands_names'][band_name])

            self.gdal_bands.update({band_name: gdal_img})

        # assign the reference band to _ref_band
        self._ref_band = self.gdal_bands[ref_band_name]

        return self.gdal_bands

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
            raise OSError('Band not found.')

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
                                                   buf_ysize=self.y_size,
                                                   resample_alg=gdal.GRA_Average).astype(dtype=np.float32) / 10000
                self.raster_bands.update({band: raster_band})

                self.invalid_mask |= raster_band == -0.9999

        return self.raster_bands

    def update_mask(self, mask):
        self.invalid_mask |= mask

        return self.invalid_mask

    def load_masks(self, product_masks_list, external_mask, mask_name, mask_valid_value=None,
                   mask_invalid_value=None):

        mask_processor = None
        if self.product == 'S2_THEIA':
            mask_processor = DWTheiaMaskProcessor(self.current_image_folder, self.x_size, self.y_size,
                                                  self.shape_file, self.temp_dir)
        elif self.product == 'LANDSAT8':
            mask_processor = DWLandsatMaskProcessor(self.current_image_folder, self.x_size, self.y_size,
                                                    self.shape_file, self.temp_dir)

        if mask_processor:
            self.update_mask(mask_processor.get_combined_masks(product_masks_list))

        if external_mask:
            mask_file = DWutils.find_file_glob(mask_name, self.current_image_folder)

            if mask_file:
                mask_ds = DWutils.read_gdal_ds(mask_file, self.shape_file, self.temp_dir)

                if mask_ds:
                    mask_array = mask_ds.ReadAsArray(buf_xsize=self.x_size, buf_ysize=self.y_size)

                    if mask_valid_value is not None:
                        print('Using external mask. Valid value = {}'.format(mask_valid_value))
                        self.update_mask(mask_array != mask_valid_value)
                    elif mask_invalid_value is not None:
                        print('Using external mask. Invalid value = {}'.format(mask_invalid_value))
                        self.update_mask(mask_array == mask_invalid_value)


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
        self._area_name = area_name

        # initialize other objects variables
        self._temp_dir = None
        self.base_name = None
        self.geo_transform = None
        self.projection = None
        self.output_folder = None

        return

    def set_output_folder(self, image_name, geo_transform, projection):
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

        if '_' in image_name:
            base_name = product_name + '-' + image_name.split('_')[1]
        elif '-' in image_name:
            base_name = product_name + '-' + image_name.split('-')[1]
        else:
            base_name = product_name + '-' + image_name

        return base_name


    @staticmethod
    def create_output_folder(output_folder, image_name, area_name):
        if not area_name:
            output_folder = output_folder.joinpath(image_name)
        else:
            output_folder = output_folder.joinpath(area_name).joinpath(image_name)

        output_folder.mkdir(parents=True, exist_ok=True)

        return output_folder

    def save_array(self, array, name, opt_relative_path=None, no_data_value=0):

        if opt_relative_path:
            filename = self.output_folder.joinpath(opt_relative_path)
            filename.mkdir(exist_ok=True)
        else:
            filename = self.output_folder

        filename = filename.joinpath(name + '.tif').as_posix()

        DWutils.array2raster(filename, array, self.geo_transform, self.projection, no_data_value)

        return filename

    def save_rgb_array(self, red, green, blue, name, opt_relative_path=None):

        if opt_relative_path:
            filename = self.output_folder.joinpath(opt_relative_path)
            filename.mkdir(exist_ok=True)
        else:
            filename = self.output_folder

        filename = filename.joinpath(name + '.tif').as_posix()

        DWutils.array2rgb_raster(filename, red, green, blue, self.geo_transform, self.projection)

        return filename

    @property
    def area_name(self):
        if self._area_name:
            return self._area_name
        else:
            return ''

    @property
    def temp_dir(self):

        if not self._temp_dir:
            self._temp_dir = self.output_folder / 'temp_dir'
            self._temp_dir.mkdir(exist_ok=True)

        return self._temp_dir


class DWLandsatMaskProcessor:

    LandsatMaskDict = {'fill': 1 << 0,
                       'clear': 1 << 1,
                       'water':  1 << 2,
                       'cloud_shadow': 1 << 3,
                       'snow': 1 << 4,
                       'cloud': 1 << 5,
                       'cloud_conf1': 1 << 6,
                       'cloud_conf2': 1 << 7,
                       'cirrus_conf1': 1 << 8,
                       'cirrus_conf2': 1 << 9,
                       'terrain_occlusion': 1 << 10
                       }

    def __init__(self, base_folder, x_size, y_size, shape_file=None, temp_dir=None):

        self.x_size = x_size
        self.y_size = y_size

        self.masks_folder = base_folder

        self.mask = self.open_mask(shape_file, temp_dir)

    def open_mask(self, shape_file, temp_dir):

        gdal_mask = self.open_gdal_masks(shape_file, temp_dir)

        raster_mask = gdal_mask.ReadAsArray(buf_xsize=self.x_size, buf_ysize=self.y_size)

        return raster_mask

    def open_gdal_masks(self, shape_file, temp_dir):

        mask_file = [file for file in self.masks_folder.glob('*pixel_qa.tif')][0]
        gdal_mask = gdal.Open(mask_file.as_posix())

        if shape_file:

            opt = gdal.WarpOptions(cutlineDSName=shape_file, cropToCutline=True,
                                   srcNodata=-9999, dstNodata=-9999, outputType=gdal.GDT_Int16)

            dest_name = (temp_dir / 'qa_cliped').as_posix()
            clipped_mask_ds = gdal.Warp(destNameOrDestDS=dest_name,
                                        srcDSOrSrcDSTab=gdal_mask,
                                        options=opt)
            clipped_mask_ds.FlushCache()
            gdal_mask = clipped_mask_ds

        return gdal_mask

    def get_combined_masks(self, masks_list):

        bitmask = 0
        for mask_key in masks_list:
            bitmask |= self.LandsatMaskDict[mask_key]

        combined_mask = np.bitwise_and(self.mask, bitmask) != 0

        return combined_mask

class DWTheiaMaskProcessor:

    TheiaMaskDict = {'CLM': '*_CLM_R1.tif',
                     'EDG': '*_EDG_R1.tif',
                     'MG2': '*_MG2_R1.tif',
                     'SAT1': '*_SAT_R1.tif',
                     'SAT2': '*_SAT_R2.tif'}

    TheiaCLMDict = {'clm_all_clouds_and_shadows': 1 << 0,
                    'clm_all_clouds': 1 << 1,
                    'clm_clouds_blue_band': 1 << 2,
                    'clm_clouds_multi_temporal': 1 << 3,
                    'clm_thin_clouds': 1 << 4,
                    'clm_cloud_shadows': 1 << 5,
                    'clm_other_shadows': 1 << 6,
                    'clm_high_clouds': 1 << 7}

    TheiaMG2Dict = {'mg2_water': 1 << 0,
                    'mg2_all_clouds': 1 << 1,
                    'mg2_snow': 1 << 2,
                    'mg2_cloud_shadows': 1 << 3,
                    'mg2_other_shadows': 1 << 4,
                    'mg2_terrain_mask': 1 << 5,
                    'mg2_sun_too_low': 1 << 6,
                    'mg2_sun_tangent': 1 << 7}

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

    def get_combined_masks(self, masks_list):

        cloud_bitmask = 0
        mg2_bitmask = 0

        # First, create a bitmask to identify the bits of the mask layer to be masked out
        for mask_key in masks_list:
            if 'clm' in mask_key:
                cloud_bitmask |= self.TheiaCLMDict[mask_key]

            if 'mg2' in mask_key:
                mg2_bitmask |= self.TheiaMG2Dict[mask_key]

        cloud_mask = np.bitwise_and(self.masks['CLM'], cloud_bitmask) != 0
        mg2_mask = np.bitwise_and(self.masks['MG2'], mg2_bitmask) != 0

        # First, create a cloud bitmask for all the clouds to be detected
        # cloud_bitmask = self.TheiaCLMDict['all_clouds_and_shadows'] | \
        #                 self.TheiaCLMDict['high_clouds'] | \
        #                 self.TheiaCLMDict['thin_clouds']

        # cloud_mask = np.bitwise_and(self.masks['CLM'], theia_cloud_mask['all_clouds_and_shadows']) != 0

        # No data mask
        edg_mask = self.masks['EDG'] != 0

        # Saturation SAT
        sat_mask = (self.masks['SAT1'] != 0) | (self.masks['SAT2'] != 0)

        # MG2 masks out snow and other shadows
        # mg2_mask = np.bitwise_and(self.masks['MG2'],
        #                           self.TheiaMG2Dict['snow'] |
        #                           self.TheiaMG2Dict['other_shadows'] |
        #                           self.TheiaMG2Dict['terrain_mask']) != 0

        return cloud_mask | edg_mask | sat_mask | mg2_mask
        # return edg_mask

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

