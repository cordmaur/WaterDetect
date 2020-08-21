from DWInputOutput import DWSaver, DWLoader
from DWCommon import DWConfig, DWutils
import DWImage
import numpy as np
from PyPDF2 import PdfFileMerger
import os
import DWInversion
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, RobustScaler
from osgeo import gdal

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
        self.loader = DWLoader(input_folder, shape_file, product, ref_band=self.config.reference_band)

        # Create a saver object
        self.saver = DWSaver(output_folder, product, self.loader.area_name)

        # if there is an inversion, create an instance of Algorithms class, None otherwise
        self.inversion_algos = DWInversion.DWInversionAlgos() if self.config.inversion else None

        return

    def necessary_bands(self, include_rgb):
        """
        Return all the necessary bands, based on the bands used for the graphics and the clustering
        :param include_rgb: Specifies if RGB bands are necessary for creating composite, for example
        :return: All necessary bands in a list
        """

        # initialize with basic bands for MNDWI and NDWI and MBWI
        necessary_bands = {'Green', 'Red', 'Blue', 'Nir', 'Mir2', 'Mir', 'RedEdg1', 'RedEdg2',
                           self.config.reference_band}

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

    def calc_m_nd_index(self, index_name, band1, band2, band3, band4, save_index=False):
        """
        Calculates a modified normalized difference index, adds it to the bands dict and update the mask in loader
        :param save_index: Inidicate if we should save this index to a raster image
        :param index_name: name of index to be used as key in the dictionary
        :param band1: first band to be used in the normalized difference
        :param band2: second band to be used in the normalized difference
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
        Calculates the Multiband Water Index and adds it to the bands dictionary
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

        mbwi[~mask] = RobustScaler(copy=False).fit_transform(mbwi[~mask].reshape(-1,1)).reshape(-1)
        # mbwi[~mask] = QuantileTransformer(copy=False).fit_transform(mbwi[~mask].reshape(-1,1)).reshape(-1)
        mbwi[~mask] = MinMaxScaler(feature_range=(-1, 1), copy=False).fit_transform(mbwi[~mask].reshape(-1,1)).reshape(-1)


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

    def list_param(self):
        """
        :return: nl: new list of the parameter separated by a "-"
                len_list: number of parameter
                list_param: list of string
        """
        list_param = []
        if ',' in self.config.parameter:
            list_param = self.config.parameter.split(',')
        else:
            list_param.append(self.config.parameter)

        len_list = len(list_param)
        nl = '-'.join(list_param)

        return nl, len_list, list_param

    def run(self):
        """
        Loop through all directories in input folder, extract water pixels and save results to output folder
        :return: None
        """

        # initialize the detect water instance variable with None
        dw_image = None

        # if pdf_report is true, creates a FileMerger to assembly the FullReport
        pdf_merger = PdfFileMerger() if self.config.pdf_reports else None

        #list of parameters
        liste_parametre, nb_param, list_param = self.list_param()

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

                # create a composite R G B in the output folder
                if self.config.create_composite or self.config.pdf_reports:
                    composite_name = DWutils.create_composite(image.gdal_bands, self.saver.output_folder,
                                                              self.config.pdf_reports)
                else:
                    composite_name = None

                # calc the sun glint rejection using the angle Tetag between vectors pointing in the surface-to-satellite
                #  and specular reflection directions
                DWutils.check_path(
                    self.loader.current_image_folder.as_posix() + '/' + image.current_image_name + '_MTD_ALL.xml')

                DWutils.extract_angles_from_xml(
                    self.loader.current_image_folder.as_posix() + '/' + image.current_image_name + '_MTD_ALL.xml')


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

                # if the method is average_results, the loop through bands_combinations will be done in DWImage module
                if self.config.average_results:
                    try:
                        print('Calculating water mask considering the average for these combinations:')
                        print(self.config.clustering_bands)

                        # Create a file merger for this report
                        if self.config.pdf_reports:
                            pdf_merger_image = PdfFileMerger()
                            pdf_merger_image.append(composite_name + '.pdf')
                        else:
                            pdf_merger_image = None

                        dw_image = self.create_water_mask(self.config.clustering_bands, pdf_merger_image)

                        # calc the inversion parameter and save it to self.rasterbands in the dictionary
                        if self.config.inversion:
                            self.calc_inversion_parameter(dw_image, pdf_merger_image)

                        # save the graphs
                        if self.config.plot_graphs:
                            self.save_graphs(dw_image, pdf_merger_image)

                        # append the pdf report of this image
                        if self.config.pdf_reports:
                            pdf_merger.append(self.save_report('ImageReport' + '_' + dw_image.product_name + '_' +
                                                               self.config.parameter,
                                                               pdf_merger_image,
                                                               self.saver.output_folder))
                    except Exception as err:
                        print('**** ERROR DURING AVERAGE CLUSTERING ****')
                        # todo: should we close the pdf merger in case of error?
                        print(err)
                    pass

                # Otherwise, loop through the bands combinations to make the clusters
                else:
                    for band_combination in self.config.clustering_bands:
                        try:
                            print('Calculating clusters for the following combination of bands:')
                            print(band_combination)

                            # if pdf_reports, create a FileMerger for this specific band combination
                            if self.config.pdf_reports:
                                pdf_merger_image = PdfFileMerger()
                                pdf_merger_image.append(composite_name + '.pdf')
                            else:
                                pdf_merger_image = None

                            # create a dw_image object with the water mask and all the results
                            dw_image = self.create_water_mask(band_combination, pdf_merger_image)

                            # calc the inversion parameter and save it to self.rasterbands in the dictionary
                            if self.config.inversion:
                                if nb_param > 1:
                                    self.calc_inversion_multiparameter(dw_image, pdf_merger_image, list_param)
                                else:
                                    self.calc_inversion_parameter(dw_image, pdf_merger_image)

                            # save the graphs
                            if self.config.plot_graphs:
                                self.save_graphs(dw_image, pdf_merger_image)

                            # append the pdf report of this image
                            if self.config.pdf_reports:
                                pdf_merger.append(self.save_report('ImageReport' + '_' + dw_image.product_name + '_' +
                                                                   self.config.parameter,
                                                                   pdf_merger_image,
                                                                   self.saver.output_folder))
                        except Exception as err:
                            print('**** ERROR DURING CLUSTERING ****')
                            # todo: should we close the pdf merger in case of error?
                            print(err)

            except Exception as err:
                print('****** ERROR ********')
                print(err)

        if pdf_merger is not None and dw_image is not None:
            if len(self.config.clustering_bands) == 1:
                report_name = 'FullReport_' + dw_image.product_name + '_' + self.config.parameter
            else:
                report_name = 'FullReport_' + self.config.parameter

            self.save_report(report_name, pdf_merger, self.saver.base_output_folder.joinpath(self.saver.area_name))

        return

    # save the report and return the full path as posix
    def save_report(self, report_name, pdf_merger, folder):

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
        dw_image = DWImage.DWImageClustering(self.loader.raster_bands, band_combination,
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
                                                                burn_in_array=dw_image.water_mask,
                                                                color=(0, 0, 1),
                                                                fade=1,
                                                                opt_relative_path=dw_image.product_name,
                                                                valid_value=1))

        return dw_image

    def calc_inversion_parameter(self, dw_image, pdf_merger_image):
        """
        Calculate the parameter in config.parameter and saves it to the dictionary of bands.
        This will make it easier to make graphs correlating any band with the parameter.
        Also, checks if there are reports, then add the parameter to it.
        :return: The parameter matrix
        """

        # POR ENQUANTO BASTA PASSARMOS O DICIONÁRIO DE BANDAS E O PRODUTO PARA TODOS

        # initialize the parameter with None
        parameter = None

        if self.config.parameter == 'turb-dogliotti':
            parameter = self.inversion_algos.turb_Dogliotti(self.loader.raster_bands['Red'],
                                                            self.loader.raster_bands['Nir'])
        elif self.config.parameter == 'spm-get':
            parameter = self.inversion_algos.SPM_GET(self.loader.raster_bands['Red'],
                                                     self.loader.raster_bands['Nir'],
                                                     self.loader.product)

        elif self.config.parameter == 'chl_lins':
            parameter = self.inversion_algos.chl_lins(self.loader.raster_bands['Red'],
                                                      self.loader.raster_bands['RedEdg1'])

        elif self.config.parameter == 'aCDOM-brezonik':
            parameter = self.inversion_algos.aCDOM_brezonik(self.loader.raster_bands['Red'],
                                                            self.loader.raster_bands['RedEdg2'],
                                                            self.loader.product)

        elif self.config.parameter == 'chl_giteslon':
            parameter = self.inversion_algos.chl_giteslon(self.loader.raster_bands['Red'],
                                                          self.loader.raster_bands['RedEdg1'],
                                                          self.loader.raster_bands['RedEdg2'])

        if parameter is not None:
            # clear the parameters array and apply the Water mask, with no_data_values
            parameter = DWutils.apply_mask(parameter, ~dw_image.water_mask, -9999)

            # save the calculated parameter
            self.saver.save_array(parameter, self.config.parameter, no_data_value=-9999)

            if pdf_merger_image is not None:

                max_value, min_value = self.calc_param_limits(parameter)

                pdf_merger_image.append(self.create_colorbar_pdf(product_name='colorbar_' + self.config.parameter,
                                                                 colormap=self.config.colormap,
                                                                 min_value=min_value,
                                                                 max_value=max_value))

                pdf_merger_image.append(self.create_rgb_burn_in_pdf(product_name=self.config.parameter,
                                                                    burn_in_array=parameter,
                                                                    color=None,
                                                                    fade=0.8,
                                                                    min_value=min_value,
                                                                    max_value=max_value,
                                                                    opt_relative_path=None,
                                                                    colormap=self.config.colormap,
                                                                    uniform_distribution=self.config.uniform_distribution,
                                                                    no_data_value=-9999))

    def calc_inversion_multiparameter(self, dw_image, pdf_merger_image, list_param):
        """
        Calculate the parameters in config.parameter and save it to the dictionary of bands.
        This will make it easier to make graphs correlating any band with the parameter.
        Also, checks if there are reports, then add the parameter to it.
        :return: The parameter matrix
        """

        # initialize the parameter with None
        parameter = None

        mask = self.loader.invalid_mask

        band = []
        red = self.loader.raster_bands['Red']
        gray = np.dot(red, 1 / 255)
        # rred = np.dot(red, 1000)

        self.saver.save_array(gray, 'red.tif', self.saver.output_folder, no_data_value=-9999)
        band = [red]

        band.append(dw_image.water_mask)

        for i in range(0, len(list_param)):
            if list_param[i] == 'turb-dogliotti':
                Red, Nir = DWutils.remove_negatives(self.loader.raster_bands['Red'],
                                                    self.loader.raster_bands['Nir'], mask)

                parameter = self.inversion_algos.turb_Dogliotti(Red, Nir)

            elif list_param[i] == 'spm-get':
                Red, Nir = DWutils.remove_negatives(self.loader.raster_bands['Red'],
                                                    self.loader.raster_bands['Nir'], mask)

                parameter = self.inversion_algos.SPM_GET(Red, Nir,
                                                         self.loader.product)

            elif list_param[i] == 'chl-lins':
                Red, RedEdg1 = DWutils.remove_negatives(self.loader.raster_bands['Red'],
                                                        self.loader.raster_bands['RedEdg1'],
                                                        mask)

                parameter = self.inversion_algos.chl_lins(Red, RedEdg1)

            elif list_param[i] == 'aCDOM-brezonik':
                Red, RedEdg2 = DWutils.remove_negatives(self.loader.raster_bands['Red'],
                                                        self.loader.raster_bands['RedEdg2'],
                                                        mask)

                parameter = self.inversion_algos.aCDOM_brezonik(Red, RedEdg2,
                                                                self.loader.product)

            elif list_param[i] == 'chl-giteslon':
                # TODO: adapt remove negatives function so it can have more than 2 input bands
                # Red, Nir = DWutils.remove_negatives(self.loader.raster_bands['Red'],
                #                                     self.loader.raster_bands['Nir'], mask)

                parameter = self.inversion_algos.chl_giteslon(self.loader.raster_bands['Red'],
                                                              self.loader.raster_bands['RedEdg1'],
                                                              self.loader.raster_bands['RedEdg2'])

            if parameter is not None:
                # print(parameter.shape)
                # clear the parameters array and apply the Water mask, with no_data_values

                param = DWutils.apply_mask(parameter,
                                           ~(np.where(dw_image.water_mask == 255, 0, dw_image.water_mask).astype(
                                               bool)),
                                           -9999)
                # save the calculated parameter
                self.saver.save_array(param, list_param[i], no_data_value=-9999)
                band.append(param)

                if pdf_merger_image is not None:
                    max_value, min_value = self.calc_param_limits(param)

                    pdf_merger_image.append(self.create_colorbar_pdf(product_name='colorbar_' + list_param[i],
                                                                     colormap=self.config.colormap,
                                                                     min_value=min_value,
                                                                     max_value=max_value))

                    pdf_merger_image.append(self.create_rgb_burn_in_pdf(product_name=list_param[i],
                                                                        burn_in_array=param,
                                                                        color=None,
                                                                        fade=0.8,
                                                                        min_value=min_value,
                                                                        max_value=max_value,
                                                                        opt_relative_path=None,
                                                                        colormap=self.config.colormap,
                                                                        uniform_distribution=self.config.uniform_distribution,
                                                                        no_data_value=-9999))

        # stack arrays
        stack = np.array(band)
        # create a tif with all bands given as parameter
        self.saver.save_multiband(stack, "multiband_" + self.list_param()[0], self.saver.output_folder,
                                  no_data_value=-9999)

    def calc_param_limits(self, parameter, no_data_value=-9999):

        valid = parameter[parameter != no_data_value]
        # min_value = np.percentile(valid, 1) if self.config.min_param_value is None else self.config.min_param_value
        min_value = np.quantile(valid, 0.25) if self.config.min_param_value is None else self.config.min_param_value
        # max_value = np.percentile(valid, 96) if self.config.max_param_value is None else self.config.max_param_value
        max_value = np.quantile(valid, 0.75) if self.config.max_param_value is None else self.config.max_param_value
        return max_value * 1.1, min_value * 0.8

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

    def create_rgb_burn_in_pdf(self, product_name, burn_in_array, color=None, min_value=None, max_value=None,
                               fade=None, opt_relative_path=None, colormap='viridis', uniform_distribution=False,
                               no_data_value=0, valid_value=1):

        # create the RGB burn in image
        red, green, blue = DWutils.rgb_burn_in(red=self.loader.raster_bands['Red'],
                                               green=self.loader.raster_bands['Green'],
                                               blue=self.loader.raster_bands['Blue'],
                                               burn_in_array=burn_in_array,
                                               color=color,
                                               min_value=min_value,
                                               max_value=max_value,
                                               colormap=colormap,
                                               fade=fade,
                                               uniform_distribution=False,
                                               no_data_value=no_data_value,
                                               valid_value=valid_value)
        print('-------------------------------TEST PDF---------------------------')
        # save the RGB auxiliary tif and gets the full path filename
        filename = self.saver.save_rgb_array(red=red * 10000,
                                             green=green * 10000,
                                             blue=blue * 10000,
                                             name=product_name+'_rgb',
                                             opt_relative_path=opt_relative_path)

        new_filename = filename[:-4] + '.pdf'
        translate = 'gdal_translate -outsize 600 0 -ot Byte -scale 0 2000 -of pdf ' + filename + ' ' + new_filename
        os.system(translate)

        return new_filename

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
        self.saver.save_array(image.invalid_mask, image.current_image_name + '_invalid_mask', dtype=gdal.GDT_Byte)

        return
