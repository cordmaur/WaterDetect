# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
from waterdetect.InputOutput import DWSaver, DWLoader
from waterdetect.Common import DWConfig, DWutils, gdal
from waterdetect.Image import DWImageClustering
from sklearn.metrics import accuracy_score, jaccard_similarity_score
import numpy as np
from PyPDF2 import PdfFileMerger
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os

"""
Author: Mauricio Cordeiro
"""


class DWWaterDetect:

    def __init__(self, input_folder, output_folder, shape_file, product, config_file, pekel=None, single_mode=False):

        # Create the Configuration object.
        # It loads the configuration file (WaterDetect.ini) and holds all the defaults if missing parameters
        self.config = DWConfig(config_file=config_file)

        # Create a Loader for the product
        self.loader = DWLoader(input_folder, shape_file, product, ref_band=self.config.reference_band,
                               single_mode=single_mode)

        # Create a saver object
        self.saver = DWSaver(output_folder, product, self.loader.area_name)

        self.single_mode = single_mode
        self.pekel = pekel

        return

    @property
    def water_mask(self):
        if hasattr(self, 'dw_image'):
            return self.dw_image.water_mask
        else:
            return None

    @property
    def cluster_matrix(self):
        if hasattr(self, 'dw_image'):
            return self.dw_image.cluster_matrix
        else:
            return None

    def __repr__(self):
        if self.water_mask is not None:
            return f'WaterDetect object with water mask and clustering results (use .water_mask ' \
                   f'or .cluster_matrix to access them)'
        else:
            return f'WaterDetect object with {len(self.loader)} images to process. \n' \
                   f'Input folder:{self.loader.input_folder}'

    def necessary_bands(self, include_rgb):
        """
        Return all the necessary bands, based on the bands used for the graphics and the clustering
        :param include_rgb: Specifies if RGB bands are necessary for creating composite, for example
        :return: All necessary bands in a list
        """

        # initialize with basic bands for MNDWI and NDWI and MBWI
        necessary_bands = {'Green', 'Red', 'Blue', 'Nir', 'Mir2', 'Mir', self.config.reference_band}

        if include_rgb:
            necessary_bands = necessary_bands.union({'Red', 'Green', 'Blue'})

        bands_cluster_set = [item for sublist in self.config.clustering_bands for item in sublist]
        bands_graphs_set = [item for sublist in self.config.graphs_bands for item in sublist]

        necessary_bands = necessary_bands.union(bands_cluster_set).union(bands_graphs_set)

        return list(necessary_bands)

    def calc_nd_index(self, index_name, band1, band2, save_index=False):
        """
        Calculates a normalized difference index, adds it to the bands dict and update the mask in loader
        :param save_index: Indicate if we should save this index to a raster image
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

    def calc_m_nd_index(self, index_name, band1, band2, band3, band4, save_index=False):
        """
        Calculates a modified normalized difference index, adds it to the bands dict and update the mask in loader.
        Proposed by Dhalton. It uses the maximum of visible and the minimum of Nir/Swir
        :param save_index: Inidicate if we should save this index to a raster image
        :param index_name: name of index to be used as key in the dictionary
        :param band1: first band to be used in the normalized difference
        :param band2: second option for the first band to be used in the normalized difference
        :param band3: second band to be used in the normalized difference
        :param band4: second option for the second band to be used in the normalized difference
        :return: index array
        """

        first = np.where(band1 >= band2, band1, band2)
        second = np.where(band3 <= band4, band3, band4)

        index, mask = DWutils.calc_normalized_difference(first, second, self.loader.invalid_mask)
        self.loader.update_mask(mask)

        self.loader.raster_bands.update({index_name: index})

        if save_index:
            self.saver.save_array(index, self.loader.current_image_name + '_' + index_name, no_data_value=-9999)

        return index

    def calc_mbwi(self, bands, factor=3, save_index=False):
        """
        Calculates the Multi band Water Index and adds it to the bands dictionary
        :param save_index: Inform if the index should be saved as array in the output folder
        :param bands: Bands dictionary with the raster bands
        :param factor: Factor to multiply the Green band as proposed in the original paper
        :return: mbwi array
        """

        mask = self.loader.invalid_mask

        # changement for negative SRE values scene
        min_cte = np.min([np.min(bands['Green'][~mask]), np.min(bands['Red'][~mask]),
                          np.min(bands['Nir'][~mask]), np.min(bands['Mir'][~mask]), np.min(bands['Mir2'][~mask])])

        if min_cte <= 0:
            min_cte = -min_cte + 0.001
        else:
            min_cte = 0

        mbwi = factor * (bands['Green']+min_cte) - (bands['Red']+min_cte) - (bands['Nir']+min_cte)\
            - (bands['Mir']+min_cte) - (bands['Mir2']+min_cte)

        mbwi[~mask] = RobustScaler(copy=False).fit_transform(mbwi[~mask].reshape(-1, 1)).reshape(-1)
        mbwi[~mask] = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(mbwi[~mask].reshape(-1, 1))\
            .reshape(-1)

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
        awei = bands['Blue'] + 2.5*bands['Green'] - 1.5*(bands['Nir'] + bands['Mir']) - 0.25*bands['Mir2']

        awei = RobustScaler(copy=False).fit_transform(awei)
        awei = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(awei)

        mask = np.isinf(awei) | np.isnan(awei) | self.loader.invalid_mask
        awei = np.ma.array(awei, mask=mask, fill_value=-9999)

        self.loader.update_mask(mask)

        bands.update({'awei': awei.filled()})

        if save_index:
            self.saver.save_array(awei.filled(), self.loader.current_image_name + '_awei', no_data_value=-9999)

        return awei.filled()

    @classmethod
    def run_single(cls, image_folder, output_folder=None, shape_file=None, product='S2_THEIA', config_file=None,
                   post_callback=None):
        """
        Run the detection algorithm for one image and one combination only.
        The input folder should be the folder of the unzipped satellite image.
        :return: instance of DWImageClustering  with mask and clustering results
        """
        wd = cls(input_folder=image_folder,
                 output_folder=output_folder,
                 shape_file=shape_file,
                 product=product,
                 config_file=config_file,
                 single_mode=True)

        wd._detect_water(post_callback=post_callback)

        return wd

    @classmethod
    def run_batch(cls, input_folder, output_folder, shape_file=None, product='S2_THEIA',
                  config_file=None, pekel=None, post_callback=None):
        """
        Run batch is intended for multi processing of various images and bands combinations.
        It Loops through all unzipped images in input folder, extract water pixels and save results to output folder
        ATT: The input folder is not a satellite image itself. It should be the parent folder containing all images.
        For single detection, use run() method.
        :return: None
        """
        wd = cls(input_folder=input_folder,
                 output_folder=output_folder,
                 shape_file=shape_file,
                 product=product,
                 config_file=config_file,
                 pekel=pekel,
                 single_mode=False)

        wd._detect_water(post_callback=post_callback)

        return

    def _detect_water(self, post_callback=None):
        """
        Run batch is intended for multi processing of various images and bands combinations.
        It Loops through all unzipped images in input folder, extract water pixels and save results to output folder
        ATT: The input folder is not a satellite image itself. It should be the parent folder containing all images.
        For single detection, use run() method.
        :return: None
        """

        # initialize the detect water instance variable with None
        dw_image = None

        # if pdf_report is true, creates a FileMerger to assembly the FullReport
        pdf_merger = PdfFileMerger() if self.config.pdf_reports else None

        # Iterate through the loader. Each image is a folder in the input directory.
        for image in self.loader:

            # Wrap the clustering loop into a try_catch to avoid single image problems to stop processing
            try:
                # prepare the saver with output folder and transformations for this image
                self.saver.set_output_folder(image.current_image_name, image.geo_transform, image.projection)

                # if there is a shape_file specified, clip necessary bands and then update the output projection
                if image.shape_file:
                    image.clip_bands(self.necessary_bands(self.config.create_composite), self.config.reference_band,
                                     self.saver.temp_dir)
                    self.saver.update_geo_transform(image.geo_transform, image.projection)

                # Load necessary bands in memory as a dictionary of names (keys) and arrays (Values)
                image.load_raster_bands(self.necessary_bands(include_rgb=False))

                # load the masks specified in the config (internal masks for theia or landsat) and the external tif mask
                image.load_masks(self.config.get_masks_list(image.product),
                                 self.config.external_mask,
                                 self.config.mask_name,
                                 self.config.mask_valid_value,
                                 self.config.mask_invalid_value)

                # Test if there is enough valid pixels in the clipped images
                if (np.count_nonzero(image.invalid_mask) / image.invalid_mask.size) > self.config.maximum_invalid:
                    print('Not enough valid pixels in the image area')
                    continue

                # calc the necessary indices and update the image's mask
                self.calc_indexes(image, indexes_list=['mndwi', 'ndwi', 'mbwi'], save_index=self.config.save_indices)

                # create a composite R G B in the output folder
                if self.config.create_composite or self.config.pdf_reports:
                    composite_name = DWutils.create_composite(image.gdal_bands, self.saver.output_folder,
                                                              self.config.pdf_reports, self.config.pdf_resolution)

                else:
                    composite_name = None

                # if the method is average_results, the loop through bands_combinations will be done in DWImage module
                if self.config.average_results:
                    print('Calculating water mask considering the average of combinations.')
                    clustering_bands = [self.config.clustering_bands]

                else:
                    if self.single_mode:
                        print('Calculating water mask in single mode. Just the first band_combination is processed')
                        clustering_bands = [self.config.clustering_bands[0]]
                    else:
                        clustering_bands = self.config.clustering_bands

                # loop through the bands combinations to make the clusters
                for band_combination in clustering_bands:
                    try:
                        print('Calculating clusters for the following combination of bands:')
                        print(band_combination)

                        dw_image = self.create_mask_report(image, band_combination, composite_name,
                                                           pdf_merger, post_callback)

                    except Exception as err:
                        print('**** ERROR DURING CLUSTERING ****')
                        # todo: should we close the pdf merger in case of error?
                        print(err)

            except Exception as err:
                print('****** ERROR ********')
                print(err)

        if pdf_merger is not None and dw_image is not None:
            if len(self.config.clustering_bands) == 1:
                report_name = 'FullReport_' + dw_image.product_name
            else:
                report_name = 'FullReport'

            self.save_report(report_name, pdf_merger, self.saver.base_output_folder.joinpath(self.saver.area_name))

        self.dw_image = dw_image
        return dw_image

    def test_pekel(self, image, dw_image, pdf_merger_image):

        water_mask = dw_image.water_mask.copy()
        pekel_mask = gdal.Open(self.pekel).ReadAsArray(buf_xsize=image.x_size,
                                                       buf_ysize=image.y_size)

        # join invalid mask with pekels invalid
        invalid_mask = image.invalid_mask.astype('bool') | (pekel_mask == 255)

        pekel_mask = pekel_mask[~invalid_mask]
        water_mask = water_mask[~invalid_mask]

        # accuracy_score(pekel_mask > pekel_threshold, water_mask == 1)
        result = jaccard_similarity_score(pekel_mask > self.config.pekel_water, water_mask == 1) * 100

        # save result to the pdf_merger
        pdf_name = os.path.join(self.saver.output_folder, 'pekel_test.pdf')

        # write the resulting text
        if result > self.config.pekel_accuracy:
            text = f'PEKEL TEST - OK \nAccuracy Threshold: {self.config.pekel_accuracy}%\nJaccard Index = {result:.3f}%'
            color = (0, 0, 0)
        else:
            text = f'*** PEKEL TEST FAILED *** \nAccuracy Threshold: {self.config.pekel_accuracy}%\n' \
                   f'Jaccard Index = {result:.3f}%'
            color = (220, 0, 0)

        pdf_merger_image.append(DWutils.write_pdf(pdf_name, text, size=(300, 100), position=(50, 15), font_color=color))
        return result

    def create_mask_report(self, image, band_combination, composite_name, pdf_merger, post_callback):
        # if pdf_reports, create a FileMerger for this specific band combination
        if self.config.pdf_reports & (pdf_merger is not None):
            pdf_merger_image = PdfFileMerger()
            pdf_merger_image.append(composite_name + '.pdf')
            # pdf_merger_image.append(invalid_mask_name)
        else:
            pdf_merger_image = None

        # create a dw_image object with the water mask and all the results
        dw_image = self.create_water_mask(band_combination, pdf_merger_image)

        # if there is a post processing callback, call it passing the mask and the pdf_merger_image
        if post_callback is not None:
            post_callback(self, dw_image=dw_image, pdf_merger=pdf_merger_image)

        # calculate the sun glint rejection and add it to the pdf report
        if self.config.calc_glint:
            self.calc_glint(image, self.saver.output_folder, pdf_merger_image)

        # Check the discrepancy of the water mask and pekel occurrence
        if self.pekel:
            self.test_pekel(image, dw_image, pdf_merger_image)

        # save the graphs
        if self.config.plot_graphs:
            self.save_graphs(dw_image, pdf_merger_image)
        # append the pdf report of this image
        if self.config.pdf_reports & (pdf_merger is not None):
            pdf_merger.append(self.save_report('ImageReport' + '_' + dw_image.product_name,
                                               pdf_merger_image,
                                               self.saver.output_folder))
        return dw_image

    # save the report and return the full path as posix
    @staticmethod
    def save_report(report_name, pdf_merger, folder):

        filename = folder.joinpath(report_name + '.pdf')

        with open(filename, 'wb') as file_obj:
            pdf_merger.write(file_obj)
        pdf_merger.close()

        return filename.as_posix()

    def save_graphs(self, dw_image, pdf_merger_image):

        # create the full path basename to plot the graphs to
        graph_basename = self.saver.output_folder.joinpath(dw_image.product_name) \
            .joinpath(self.saver.base_name + dw_image.product_name).as_posix()

        # for the PDF writer, we need to pass all the information needed for the title
        # so, we will produce the graph title = Area + Date + Product
        graph_title = self.saver.area_name + ' ' + self.saver.base_name + dw_image.product_name

        DWutils.plot_graphs(self.loader.raster_bands, self.config.graphs_bands, dw_image.cluster_matrix,
                            graph_basename, graph_title, self.loader.invalid_mask, 1000, pdf_merger_image)

    def create_water_mask(self, band_combination, pdf_merger_image):

        # create the clustering image
        dw_image = DWImageClustering(self.loader.raster_bands, band_combination,
                                     self.loader.invalid_mask, self.config)
        dw_image.run_detect_water()

        # save the water mask and the clustering results
        self.saver.save_array(dw_image.water_mask, dw_image.product_name + '_water_mask',
                              opt_relative_path=dw_image.product_name, dtype=gdal.GDT_Byte)
        self.saver.save_array(dw_image.cluster_matrix, dw_image.product_name + '_clusters',
                              opt_relative_path=dw_image.product_name, dtype=gdal.GDT_Byte)
        # unload bands

        # if there is a pdf to create, burn-in the mask into the RGB composite
        # and append it to the image merger
        if pdf_merger_image:
            pdf_merger_image.append(self.create_rgb_burn_in_pdf(dw_image.product_name + '_water_mask',
                                                                burn_in_arrays=dw_image.water_mask,
                                                                colors=(0, 0, 1),
                                                                fade=1,
                                                                opt_relative_path=dw_image.product_name,
                                                                valid_value=1))

            pdf_merger_image.append(self.create_rgb_burn_in_pdf(dw_image.product_name + '_overlay',
                                                                burn_in_arrays=[dw_image.water_mask,
                                                                                self.loader.invalid_mask],
                                                                colors=[(0, 0, 1), (1, 0, 0)],
                                                                fade=1,
                                                                opt_relative_path=dw_image.product_name,
                                                                valid_value=1,
                                                                transps=[0, 0.7]))

        return dw_image

    def calc_glint(self, image, output_folder, pdf_merger_image):
        """
        Calculate the sun glint rejection using the angle Tetag between vectors pointing in the surface-to-satellite
        and specular reflection directions
        Also, checks if there are reports, then add the risk of glint to it.
        """
        xml = str(self.loader.metadata)
        # check the path of the metadata file
        DWutils.check_path(xml)
        # extract angles from the metadata and make the glint calculation from it
        glint = DWutils.extract_angles_from_xml(xml)

        # create a pdf file that indicate if there is glint on the image and add it to the final pdf report
        DWutils.create_glint_pdf(xml, self.loader.glint_name, output_folder, glint, pdf_merger_image)

    def create_colorbar_pdf(self, product_name, colormap, min_value, max_value):

        filename = self.saver.output_folder.joinpath(product_name + '.pdf')

        p_name = product_name.split('_')
        name_param = p_name[1]

        DWutils.create_colorbar_pdf(product_name=filename,
                                    title=self.saver.area_name + ' ' + self.saver.base_name,
                                    label=name_param + ' ' + DWConfig._units[name_param],
                                    # label=self.config.parameter + ' ' + self.config.parameter_unit,
                                    colormap=colormap,
                                    min_value=min_value,
                                    max_value=max_value)

        return filename.as_posix()

    def create_rgb_burn_in_pdf(self, product_name, burn_in_arrays, colors=None, min_value=None, max_value=None,
                               fade=None, opt_relative_path=None, colormap='viridis', uniform_distribution=False,
                               no_data_value=0, valid_value=1, transps=None, bright=5.):

        red = self.loader.raster_bands['Red']*bright
        green = self.loader.raster_bands['Green']*bright
        blue = self.loader.raster_bands['Blue']*bright

        # limit the maximum brightness to 1
        red[red > 1] = 1
        green[green > 1] = 1
        blue[blue > 1] = 1

        if isinstance(burn_in_arrays, list):
            for burn_in_array, color, transp in zip(burn_in_arrays, colors, transps):
                red, green, blue = DWutils.rgb_burn_in(red=red,
                                                       green=green,
                                                       blue=blue,
                                                       burn_in_array=burn_in_array,
                                                       color=color,
                                                       min_value=min_value,
                                                       max_value=max_value,
                                                       colormap=colormap,
                                                       fade=fade,
                                                       uniform_distribution=uniform_distribution,
                                                       no_data_value=no_data_value,
                                                       valid_value=valid_value,
                                                       transp=transp)
        else:
            # create the RGB burn in image
            red, green, blue = DWutils.rgb_burn_in(red=red,
                                                   green=green,
                                                   blue=blue,
                                                   burn_in_array=burn_in_arrays,
                                                   color=colors,
                                                   min_value=min_value,
                                                   max_value=max_value,
                                                   colormap=colormap,
                                                   fade=fade,
                                                   uniform_distribution=uniform_distribution,
                                                   no_data_value=no_data_value,
                                                   valid_value=valid_value)

        # save the RGB auxiliary tif and gets the full path filename
        filename = self.saver.save_rgb_array(red=red * 10000,
                                             green=green * 10000,
                                             blue=blue * 10000,
                                             name=product_name+'_rgb',
                                             opt_relative_path=opt_relative_path)

        pdf_filename = DWutils.tif_2_pdf(filename, self.config.pdf_resolution, scale=10000)

        # remove the RGB auxiliary tif (too big to be kept)
        os.remove(filename)

        return pdf_filename

    def calc_indexes(self, image, indexes_list, save_index=False):

        raster_bands = image.raster_bands

        if 'mndwi' in indexes_list:
            # calculate the MNDWI, update the mask and saves it
            self.calc_nd_index('mndwi', raster_bands['Green'], raster_bands['Mir'], save_index=save_index)

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
        print('Saving final MASK')
        self.saver.save_array(image.invalid_mask, image.current_image_name + '_invalid_mask', dtype=gdal.GDT_Byte)

        return
