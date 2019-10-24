from DWInputOutput import DWutils, DWSaver, DWLoader, DWConfig
import DWImage
import numpy as np
import ast


class DWWaterDetect:

    # initialize the variables
    # max_invalid_pixels = 0.8  # The maximum percentage of invalid (masked) pixels to continue
    # min_mndwi = 0.0  # mndwi threshold
    # clustering = 'aglomerative'  # aglomerative, kmeans, gauss_mixture
    # classifier = 'naive_bayes'  # naive_bays, MLP, Hull, SVM
    # clip_mndwi = None  # None or mndwi value to clip false positives
    # ref_band = 'Red'
    # config = None  # Configurations loaded from WaterDetect.ini

    def __init__(self, input_folder, output_folder, shape_file, product):

        # Load the configuration file (WaterDetect.ini)
        # self.config = self.load_config_file()
        self.config = DWConfig()

        # initialize some parameters
        # bands_cluster are the bands combinations to use in the clustering algorithm
        # self.bands_cluster = [['ndwi', 'mndwi']]

        # bands_graphs are the bands combinations to generate the graphs
        # self.bands_graphs = [['Mir2', 'mndwi'], ['ndwi', 'mndwi']]

        # sets the reference band for resolution and projections
        # self.ref_band = 'Red'

        # indicate if it is necessary to create a composite output of the area
        # self.create_composite = True

        # create a Loader for the product
        self.loader = DWLoader(input_folder, shape_file, product)

        # create a saver object
        self.saver = DWSaver(output_folder, product, self.loader.area_name)

        return

    def load_mask_bands(self, image_folder):
        masks = []
        masks_folder = image_folder/'MASKS'

        if self.product == 'S2_THEIA':

            print('Retrieving THEIA masks:')
            # get 10m resolution masks
            masks.append([file for file in masks_folder.iterdir() if
                          file.suffix == '.tif' and 'R1' in file.stem and ('CLM' in file.stem or
                                                                           'EDG' in file.stem or
                                                                           'MG2' in file.stem or
                                                                           'SAT' in file.stem)])

            masks.append([file for file in masks_folder.iterdir() if
                          file.suffix == '.tif' and 'R2' in file.stem and ('CLM' in file.stem or
                                                                           'EDG' in file.stem or
                                                                           'MG2' in file.stem or
                                                                           'SAT' in file.stem)])

            for b in masks:
                print(b)

        return masks

    def necessary_bands(self, include_rgb):

        # initialize with basic bands for MNDWI and NDWI and MBWI
        necessary_bands = {'Green', 'Red', 'Nir', 'Mir', 'Mir2'}

        if include_rgb:
            necessary_bands = necessary_bands.union({'Red', 'Green', 'Blue', self.config.reference_band})

        bands_cluster_set = [item for sublist in self.config.clustering_bands for item in sublist]
        bands_graphs_set = [item for sublist in self.config.graphs_bands for item in sublist]

        necessary_bands = necessary_bands.union(bands_cluster_set).union(bands_graphs_set)

        return list(necessary_bands)

    def calc_nd_index(self, index_name, band1, band2, bands_dict=None):
        """
        Calculates a normalized difference index, adds it to the bands dict and update the mask in loader
        :param index_name: name of index to be used as key in the dictionary
        :param band1: first band to be used in the normalized difference
        :param band2: second band to be used in the normalized difference
        :param bands_dict: dictionary of the bands
        :return: index array
        """

        index, mask = DWutils.calc_normalized_difference(band1, band2)
        self.loader.update_mask(mask)

        if bands_dict:
            bands_dict.update({index_name: index})

        return index

    def calc_mbwi(self, bands, factor=3):

        mbwi = factor * bands['Green'] - bands['Red'] - bands['Nir'] - bands['Mir'] - bands['Mir2']

        mask = np.isinf(mbwi) | np.isnan(mbwi)
        mbwi = np.ma.array(mbwi, mask=mask, fill_value=-9999)

        self.loader.update_mask(mask)

        bands.update({'mbwi': mbwi.filled()})

        return mbwi.filled()

    def run(self):

        # todo: wrap everything in a try catch loop
        for image in self.loader:
            image = self.loader

            # open image into DWLoader class, passing the reference band
            image.open_current_image(ref_band_name=self.config.reference_band)

            self.saver.set_output_image(image.name, image.geo_transform, image.projection)

            # if there is a shape_file specified, make clipping of necessary bands and then update the output projection
            if image.shape_file:
                image.clip_bands(self.necessary_bands(self.config.create_composite), self.config.reference_band, self.saver.temp_dir)
                self.saver.update_geo_transform(image.geo_transform, image.projection)

            # create a composite R G B in the output folder
            if self.config.create_composite:
                DWutils.create_composite(image.gdal_bands, self.saver.output_folder)

            # Load necessary bands in memory
            raster_bands = image.load_raster_bands(self.necessary_bands(include_rgb=False))

            # todo: correct the masks
            image.load_masks()

            # calculate the MNDWI, update the mask and saves it
            mndwi = self.calc_nd_index('mndwi', raster_bands['Green'], raster_bands['Mir2'], raster_bands)
            self.saver.save_array(mndwi, image.name + '_MNDWI')

            # calculate the NDWI update the mask and saves it
            ndwi = self.calc_nd_index('ndwi', raster_bands['Green'], raster_bands['Nir'], raster_bands)
            self.saver.save_array(ndwi, image.name + '_NDWI')

            # calculate the MultiBand index using: Green, Red, Nir, Mir1, Mir2
            mbwi = self.calc_mbwi(raster_bands, factor=2)
            self.saver.save_array(mbwi, image.name + '_MBWI')

            # save the mask
            self.saver.save_array(image.invalid_mask, image.name + '_invalid_mask')

            # loop through the bands combinations to make the clusters
            for band_combination in self.config.clustering_bands:

                print('Calculating clusters for the following combination of bands:')
                print(band_combination)

                # create the clustering image
                dw_image = DWImage.DWImageClustering(raster_bands, band_combination, image.invalid_mask, self.config)
                matrice_cluster = dw_image.run_detect_water()

                # prepare the base product name based on algorithm and bands, to create the directory
                cluster_product_name = dw_image.create_product_name()

                # save the water mask and the clustering results
                self.saver.save_array(dw_image.water_mask, cluster_product_name + '_water_mask',
                                      opt_relative_path=cluster_product_name)
                self.saver.save_array(dw_image.cluster_matrix, cluster_product_name + '_clusters',
                                      opt_relative_path=cluster_product_name)

                # unload bands

                # plot the graphs specified in graph_bands
                graph_basename = self.saver.output_folder.joinpath(cluster_product_name)\
                    .joinpath(self.saver.base_name + cluster_product_name).as_posix()

                DWutils.plot_graphs(raster_bands, self.config.graphs_bands, matrice_cluster,
                                    graph_basename, image.invalid_mask, 1000)

            # Test if there is enough valid pixels in the clipped images
            # if (np.count_nonzero(image.invalid_mask) / image.invalid_mask.size) > self.max_invalid_pixels:
            #     print('Not enough valid pixels in the image area')
            #     return

            # bands = self.load_product_bands(image)
            # masks = self.load_mask_bands(image)

            # if there are bands loaded call the water detection algorithm
            # if bands:
            #     CalculateStatistics2.Treat_files(bands, masks, self.product, self.output_folder, self.shape_file)

        return

