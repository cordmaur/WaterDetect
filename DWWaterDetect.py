from DWInputOutput import DWSaver, DWLoader
from DWCommon import DWConfig, DWutils
import DWImage
import numpy as np



class DWWaterDetect:

    # initialize the variables
    # max_invalid_pixels = 0.8  # The maximum percentage of invalid (masked) pixels to continue
    # min_mndwi = 0.0  # mndwi threshold
    # clustering = 'aglomerative'  # aglomerative, kmeans, gauss_mixture
    # classifier = 'naive_bayes'  # naive_bays, MLP, Hull, SVM
    # clip_mndwi = None  # None or mndwi value to clip false positives
    # ref_band = 'Red'
    # config = None  # Configurations loaded from WaterDetect.ini

    def __init__(self, input_folder, output_folder, shape_file, product, config_file):

        # Create the Configuration object.
        # It loads the configuration file (WaterDetect.ini) and holds all the defaults if missing parameters
        self.config = DWConfig(config_file=config_file)

        # Create a Loader for the product
        self.loader = DWLoader(input_folder, shape_file, product)

        # Create a saver object
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
        """
        Return all the necessary bands, based on the bands used for the graphics and the clustering
        :param include_rgb: Specifies if RGB bands are necessary for creating composite, for example
        :return: All necessary bands in a list
        """

        # initialize with basic bands for MNDWI and NDWI and MBWI
        necessary_bands = {'Green', 'Red', 'Blue', 'Nir', 'Mir', 'Mir2', self.config.reference_band}

        if include_rgb:
            necessary_bands = necessary_bands.union({'Red', 'Green', 'Blue'})

        bands_cluster_set = [item for sublist in self.config.clustering_bands for item in sublist]
        bands_graphs_set = [item for sublist in self.config.graphs_bands for item in sublist]

        necessary_bands = necessary_bands.union(bands_cluster_set).union(bands_graphs_set)

        return list(necessary_bands)

    def calc_nd_index(self, index_name, band1, band2, save_index=False):
        """
        Calculates a normalized difference index, adds it to the bands dict and update the mask in loader
        :param save_index: Inidicate if we should save this index to a raster image
        :param index_name: name of index to be used as key in the dictionary
        :param band1: first band to be used in the normalized difference
        :param band2: second band to be used in the normalized difference
        :param bands_dict: dictionary of the bands
        :return: index array
        """

        index, mask = DWutils.calc_normalized_difference(band1, band2, self.loader.invalid_mask)
        self.loader.update_mask(mask)

        self.loader.raster_bands.update({index_name: index})

        if save_index:
            self.saver.save_array(index, self.loader.current_image_name + '_' + index_name)

        return index

    def calc_mbwi(self, bands, factor=3, save_index=False):
        """
        Calculates the Multiband Water Index and adds it to the bands dictionary
        :param bands: Bands dictionary with the raster bands
        :param factor: Factor to multiply the Green band as proposed in the original paper
        :return: mbwi array
        """
        mbwi = factor * bands['Green'] - bands['Red'] - bands['Nir'] - bands['Mir'] - bands['Mir2']

        mask = np.isinf(mbwi) | np.isnan(mbwi) | self.loader.invalid_mask
        mbwi = np.ma.array(mbwi, mask=mask, fill_value=-9999)

        self.loader.update_mask(mask)

        bands.update({'mbwi': mbwi.filled()})

        if save_index:
            self.saver.save_array(mbwi.filled(), self.loader.current_image_name + '_mbwi')

        return mbwi.filled()

    def calc_awei(self, bands, save_index=False):
        """
        Calculates the AWEI Water Index and adds it to the bands dictionary
        :param bands: Bands dictionary with the raster bands
        :return: mbwi array
        """
        awei = bands['Blue'] + 2.5*bands['Green'] - 1.5*(bands['Red'] + bands['Mir']) - 0.25*bands['Mir2']

        mask = np.isinf(awei) | np.isnan(awei) | self.loader.invalid_mask
        awei = np.ma.array(awei, mask=mask, fill_value=-9999)

        self.loader.update_mask(mask)

        bands.update({'awei': awei.filled()})


        return awei.filled()

    def run(self):

        # todo: wrap everything in a try catch loop
        for image in self.loader:
            image = self.loader

            # open image into DWLoader class, passing the reference band
            image.open_current_image(ref_band_name=self.config.reference_band)

            self.saver.set_output_folder(image.current_image_name, image.geo_transform, image.projection)

            # if there is a shape_file specified, make clipping of necessary bands and then update the output projection
            if image.shape_file:
                image.clip_bands(self.necessary_bands(self.config.create_composite), self.config.reference_band,
                                 self.saver.temp_dir)
                self.saver.update_geo_transform(image.geo_transform, image.projection)

            # create a composite R G B in the output folder
            if self.config.create_composite:
                DWutils.create_composite(image.gdal_bands, self.saver.output_folder)

            # Load necessary bands in memory
            raster_bands = image.load_raster_bands(self.necessary_bands(include_rgb=False))

            # todo: correct the masks
            image.load_masks(self.config.get_masks_list(image.product))

            # Test if there is enough valid pixels in the clipped images
            if (np.count_nonzero(image.invalid_mask) / image.invalid_mask.size) > 0.8:
                print('Not enough valid pixels in the image area')
                continue

            # calculate the MNDWI, update the mask and saves it
            self.calc_nd_index('mndwi', raster_bands['Green'], raster_bands['Mir2'], save_index=True)

            # calculate the NDWI update the mask and saves it
            self.calc_nd_index('ndwi', raster_bands['Green'], raster_bands['Nir'], save_index=True)

            # calculate the MultiBand index using: Green, Red, Nir, Mir1, Mir2
            self.calc_mbwi(raster_bands, factor=2, save_index=True)

            # calculate the MultiBand index using: Green, Red, Nir, Mir1, Mir2
            self.calc_awei(raster_bands, save_index=True)

            # calculate the MultiBand index using: Green, Red, Nir, Mir1, Mir2
            # ndvi = self.calc_nd_index('ndvi', raster_bands['Nir'], raster_bands['Red'], raster_bands)
            # self.saver.save_array(ndvi, image.name + '_NDVI')

            # save the final mask
            self.saver.save_array(image.invalid_mask, image.current_image_name + '_invalid_mask')

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

            # bands = self.load_product_bands(image)
            # masks = self.load_mask_bands(image)

            # if there are bands loaded call the water detection algorithm
            # if bands:
            #     CalculateStatistics2.Treat_files(bands, masks, self.product, self.output_folder, self.shape_file)

        return

