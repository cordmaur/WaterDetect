from DWInputOutput import DWSaver, DWLoader
from DWCommon import DWConfig, DWutils
import DWImage
import numpy as np
from PyPDF2 import PdfFileMerger
import os

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
        :return: index array
        """

        index, mask = DWutils.calc_normalized_difference(band1, band2, self.loader.invalid_mask)
        self.loader.update_mask(mask)

        self.loader.raster_bands.update({index_name: index})

        if save_index:
            self.saver.save_array(index, self.loader.current_image_name + '_' + index_name, no_data_value=-9999)

        return index

    def calc_mbwi(self, bands, factor=3, save_index=False):
        """
        Calculates the Multiband Water Index and adds it to the bands dictionary
        :param save_index: Inform if the index should be saved as array in the output folder
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
            self.saver.save_array(mbwi.filled(), self.loader.current_image_name + '_mbwi', no_data_value=-9999)

        return mbwi.filled()

    def calc_awei(self, bands, save_index=False):
        """
        Calculates the AWEI Water Index and adds it to the bands dictionary
        :param save_index: Inform if the index should be saved as array in the output folder
        :param bands: Bands dictionary with the raster bands
        :return: mbwi array
        """
        awei = bands['Blue'] + 2.5*bands['Green'] - 1.5*(bands['Red'] + bands['Mir']) - 0.25*bands['Mir2']

        mask = np.isinf(awei) | np.isnan(awei) | self.loader.invalid_mask
        awei = np.ma.array(awei, mask=mask, fill_value=-9999)

        self.loader.update_mask(mask)

        bands.update({'awei': awei.filled()})

        if save_index:
            self.saver.save_array(awei.filled(), self.loader.current_image_name + '_awei', no_data_value=-9999)

        return awei.filled()

    def run(self):

        # if pdf_report is true, creates a FileMerger to assembly the FullReport
        pdf_merger = PdfFileMerger() if self.config.pdf_reports else None

        for image in self.loader:

            # wrap the clustering loop into a try_catch to avoid single image problems
            try:
                image = self.loader

                # todo: use the output folder to create a temporary folder and simplify the clipping

                # open image into DWLoader class, passing the reference band
                image.open_current_image(ref_band_name=self.config.reference_band)

                # prepare the saver with output folder and transformations
                self.saver.set_output_folder(image.current_image_name, image.geo_transform, image.projection)

                # if there is a shape_file specified, clip necessary bands and then update the output projection
                if image.shape_file:
                    image.clip_bands(self.necessary_bands(self.config.create_composite), self.config.reference_band,
                                     self.saver.temp_dir)
                    self.saver.update_geo_transform(image.geo_transform, image.projection)

                # create a composite R G B in the output folder
                if self.config.create_composite or self.config.pdf_reports:
                    composite_name = DWutils.create_composite(image.gdal_bands, self.saver.output_folder,
                                                              self.config.pdf_reports)
                else:
                    composite_name = None

                # Load necessary bands in memory
                raster_bands = image.load_raster_bands(self.necessary_bands(include_rgb=False))

                # load the masks specified in the config
                image.load_masks(self.config.get_masks_list(image.product))

                # Test if there is enough valid pixels in the clipped images
                if (np.count_nonzero(image.invalid_mask) / image.invalid_mask.size) > 0.8:
                    print('Not enough valid pixels in the image area')
                    continue

                # calc the necessary indexes and update the image's mask
                self.calc_indexes(image, indexes_list=['mndwi', 'ndwi', 'mbwi'], save_index=True)

                ##################################################################
                self.calc_texture(image, save_texture=True)

                # print (STOP)
                ##################################################################

                # loop through the bands combinations to make the clusters
                for band_combination in self.config.clustering_bands:

                    print('Calculating clusters for the following combination of bands:')
                    print(band_combination)

                    # if there pdf_reports, create a FileMerger for this specific band combination
                    if self.config.pdf_reports:
                        pdf_merger_image = PdfFileMerger()
                        pdf_merger_image.append(composite_name + '.pdf')
                    else:
                        pdf_merger_image = None

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

                    # if there is a pdf to create, burn-in the mask into the RGB composite
                    # and append it to the image merger
                    if self.config.pdf_reports:
                        pdf_merger_image.append(self.create_rgb_burn_in_mask(dw_image))

                    # create the full path basename to plot the graphs to
                    graph_basename = self.saver.output_folder.joinpath(cluster_product_name)\
                        .joinpath(self.saver.base_name + cluster_product_name).as_posix()

                    # for the PDF writer, we need to pass all the information needed for the title
                    # so, we will produce the graph title = Area + Date + Product
                    graph_title = self.saver.area_name + ' ' + self.saver.base_name + cluster_product_name

                    DWutils.plot_graphs(raster_bands, self.config.graphs_bands, matrice_cluster,
                                        graph_basename, graph_title, image.invalid_mask, 1000, pdf_merger_image)

                    if self.config.pdf_reports:
                        report_name = 'ImageReport' + '_' + cluster_product_name + '.pdf'

                        with open(self.saver.output_folder.joinpath(report_name), 'wb') as file_obj:
                            pdf_merger_image.write(file_obj)

                        pdf_merger_image.close()

                        pdf_merger.append(self.saver.output_folder.joinpath(report_name).as_posix())

            except OSError:
                    print(OSError)

        if pdf_merger:
            if len(self.config.clustering_bands) == 1:
                report_name = 'FullReport'
                for band in self.config.clustering_bands[0]:
                    report_name += '_' + band
                report_name += '.pdf'
            else:
                report_name = 'FullReport.pdf'
            with open(self.saver.base_output_folder.joinpath(self.saver.area_name).
                      joinpath(report_name), 'wb') as file_obj:
                pdf_merger.write(file_obj)

        return

    def create_rgb_burn_in_mask(self, dw_image):

        cluster_product_name = dw_image.create_product_name()
        red = np.copy(dw_image.bands['Red'])
        red[dw_image.water_mask == 1] = 0
        green = np.copy(dw_image.bands['Green'])
        green[dw_image.water_mask == 1] = 0
        blue = np.copy(dw_image.bands['Blue'])
        blue[dw_image.water_mask == 1] = np.max(dw_image.bands['Blue'])

        filename = self.saver.save_rgb_array(red * 10000, green * 10000,
                                             blue * 10000, cluster_product_name + '_mask',
                                             opt_relative_path=cluster_product_name)

        new_filename = filename[:-4] + '.pdf'
        translate = 'gdal_translate -outsize 600 0 -ot Byte -scale 0 2000 -of pdf ' + filename + ' ' + new_filename
        os.system(translate)

        return new_filename

    def calc_indexes(self, image, indexes_list, save_index=False):

        raster_bands = image.raster_bands

        if 'mndwi' in indexes_list:
            # calculate the MNDWI, update the mask and saves it
            self.calc_nd_index('mndwi', raster_bands['Green'], raster_bands['Mir2'], save_index=save_index)

        if 'ndwi' in indexes_list:
            # calculate the NDWI update the mask and saves it
            self.calc_nd_index('ndwi', raster_bands['Green'], raster_bands['Nir'], save_index=save_index)

        if 'ndvi' in indexes_list:
            # calculate the NDVI update the mask and saves it
            self.calc_nd_index('ndvi', raster_bands['Nir'], raster_bands['Red'], save_index=save_index)

        if 'mbwi' in indexes_list:
            # calculate the MultiBand index using: Green, Red, Nir, Mir1, Mir2
            self.calc_mbwi(raster_bands, factor=2, save_index=save_index)

        if 'awei' in indexes_list:
            # calculate the MultiBand index using: Green, Red, Nir, Mir1, Mir2
            self.calc_awei(raster_bands, save_index=save_index)

        # update the final mask
        self.saver.save_array(image.invalid_mask, image.current_image_name + '_invalid_mask')

    def sliding_window(Self, img, patch_size=5,
                       istep=1, jstep=1, scale=1.0):
        # Ni, Nj = (int(scale * s) for s in patch_size)
        Ni = patch_size
        Nj = patch_size

        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Ni, jstep):
                patch = img[i:i + Ni, j:j + Nj]
                # if scale != 1:
                #     patch = transform.resize(patch, patch_size)
                yield (i, j), patch

    def calc_texture(self, image, save_texture=False):

        texture_band = image.raster_bands['Mir2']

        std_stack = np.stack((texture_band, np.roll(texture_band, 1, 0), np.roll(texture_band, -1, 0), np.roll(texture_band, 1, 1),
                              np.roll(texture_band, -1, 1),
                              np.roll(np.roll(texture_band, 1, 0), 1, 1),
                              np.roll(np.roll(texture_band, 1, 0), -1, 1),
                              np.roll(np.roll(texture_band, -1, 0), 1, 1),
                              np.roll(np.roll(texture_band, -1, 0), -1, 1)), 2)

        std = np.std(std_stack, 2)
        # std = texture_band
        #
        # std[std > 0.05] = np.max(std[std <= 0.05])

        # std1 = image.raster_bands['mndwi'] - 50 * std
        # std2 = image.raster_bands['ndwi'] - 50 * std

        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer()

        # transform std in column to apply the qualtile transform
        norm_std = qt.fit_transform(std.reshape(-1, 1))
        norm_std = norm_std.reshape(std.shape)

        std1 = image.raster_bands['mndwi'] - norm_std
        std2 = image.raster_bands['ndwi'] - norm_std

        if save_texture:
            self.saver.save_array(norm_std, image.current_image_name + '_std', no_data_value=0)

        # image.raster_bands.update({'std': norm_std})
        image.raster_bands.update({'std1': std1})
        image.raster_bands.update({'std2': std2})

        return

        # indices, patches = zip(*self.sliding_window(nir))

        # std_dev = np.array([np.std(patch) for patch in patches])

        # the last indice holds the shape of the patches matrix
        # std = np.zeros(indices[-1])
        # std[std==0]=std_dev

        # or, using the reshape function
        # std_dev.reshape(indices[-1])

        # print(std_dev.shape)

        # self.saver.save_array(std_dev, 'nir_std_dev', no_data_value=-9999)


        # from skimage.feature import greycomatrix, greycoprops

        # glcm = greycomatrix(raster_bands['Nir'], [5], [0], 256, symmetric=True, normed=True)
        # diss = greycoprops(glcm, 'dissimilarity')[0, 0]
        # ys.append(greycoprops(glcm, 'correlation')[0, 0])

        # otsu thresholding on MNDWI just like every implementation
        # from skimage.filters import threshold_otsu
        # threshold = threshold_otsu(mndwi[mndwi > -9999])

        # from skimage.segmentation import random_walker
        #
        # mndwi[mndwi<-1] = -1
        # mndwi[mndwi>1] = 1
        # markers = np.zeros(mndwi.shape, dtype=np.uint)
        # markers[mndwi < -0.4] = 1
        # markers[mndwi > 0.4] = 2
        #
        # Run random walker algorithm
        # labels = random_walker(mndwi, markers, beta=10, mode='bf')
        # self.saver.save_array(labels, 'mndwi_walker_labels', no_data_value=-9999)

        # otsu = np.copy(mndwi)
        # otsu[otsu < threshold] = -9999
        #
        # from skimage.filters import threshold_local
        # local_threshold = threshold_local(mndwi, block_size=999)
        # self.saver.save_array(mndwi>local_threshold, 'mndwi_local_threshold.tif', no_data_value=-9999)

