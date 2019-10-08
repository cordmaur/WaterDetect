# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (C) CNES - All Rights Reserved
This file is subject to the terms and conditions defined in
file 'LICENSE.md', which is part of this source code package.

Author:         Peter KETTIG <peter.kettig@cnes.fr>
Corresponding Author : jean-michel.martinez@ird.fr, UMR GET
Project:        WaterColor, CNES
Created on:     Fri Nov 23 10:30:20 2018
"""

# from StatisticsCalculation import Processing
# from ProductFormatter import MuscateWriter
# from Preprocessing import checkDates, testParameters
# from copy import deepcopy
# from Common import ImageIO
# from Common.ImageOperations import merge, clip
# from Common import GeoCoordinates as geo
# from Common import FileSystem as fs
# from Common.createColormaps import ColormapCreator
# import os
# import numpy as np
# from StatisticsCalculation import CalculateStatistics2

import numpy as np
from DWInputOutput import DWutils, DWSaver, DWLoader
import DWImage


class WaterDetect:

    # initialize the variables
    max_invalid_pixels = 0.8  # The maximum percentage of invalid (masked) pixels to continue
    min_mndwi = 0.0  # mndwi threshold
    clustering = 'aglomerative'  # aglomerative, kmeans, gauss_mixture
    classifier = 'naive_bayes'  # naive_bays, MLP, Hull, SVM
    clip_mndwi = 0.05  # None or mndwi value to clip false positives
    ref_band = 'Red'

    def __init__(self, input_folder, output_folder, shape_file, product):

        self.bands_cluster = [['Mir2', 'mndwi'], ['ndwi', 'mndwi']]
        self.bands_graphs = [['Mir2', 'mndwi'], ['ndwi', 'mndwi']]
        self.create_composite = True

        self.loader = DWLoader(input_folder, shape_file, product)

        self.output_folder = DWutils.check_path(output_folder, is_dir=True)

        self.product = product

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

    def necessary_bands(self, ref_band):

        if self.create_composite:
            necessary_bands = {'Red', 'Green', 'Blue', 'Nir2', 'Mir2', ref_band}
        else:
            necessary_bands = set(ref_band)

        necessary_bands = necessary_bands.union(set([item for sublist in self.bands_cluster for item in sublist]))
        necessary_bands = necessary_bands.union(set([item for sublist in self.bands_graphs for item in sublist]))

        return list(necessary_bands)

    def run(self):

        # todo: colocar tudo dentro do loop em um try catch para pular as imagens com erro
        for image in self.loader:
            image = self.loader

            # open image into DWLoader class, passing the reference band
            image.open_image(ref_band_name='Red')

            saver = DWSaver(self.output_folder, image.name(), image.product,
                            image.get_geo_transform(), image.get_projection(), 'TestNewSystem')

            if image.shape_file:
                image.clip_bands(self.necessary_bands('Red'), 'Red', saver.temp_dir)

            if self.create_composite:
                DWutils.create_composite(image.gdal_bands, saver.output_folder)

            bands = image.load_raster_bands(['Green', 'Mir2', 'Nir2'])

            # calculate the MNDWI mask and saves it
            mndwi, mask = DWutils.calc_normalized_difference(bands['Green'], bands['Mir2'])
            image.update_mask(mask)
            bands.update({'mndwi': mndwi})
            saver.save_array(mndwi, 'MNDWI')

            # calculate the NDWI mask and saves it
            ndwi, mask = DWutils.calc_normalized_difference(bands['Green'], bands['Nir2'])
            image.update_mask(mask)
            bands.update({'ndwi': ndwi})

            # if bands_keys is not a list of lists, transform it
            if type(self.bands_cluster[0]) == str:
                self.bands_cluster = [self.bands_cluster]

            # loop through the bands combinations to make the clusters
            for band_combination in self.bands_cluster:

                image.load_raster_bands(band_combination)

                print('Calculating cluster for the following combination of bands:')
                print(band_combination)

                # create the clustering image
                dw_image = DWImage.DWImageClustering(bands, band_combination, image.invalid_mask, {})
                matrice_cluster = dw_image.run_detect_water()

                # prepare the base product name based on algorithm and bands, to create the directory
                product_name = dw_image.create_product_name()

                # save the water mask and the clustering results
                # saver.save_array(bands['Green'], 'water_mask', opt_relative_path=product_name)
                saver.save_array(dw_image.water_mask, 'water_mask', opt_relative_path=product_name)
                saver.save_array(dw_image.cluster_matrix, 'clusters', opt_relative_path=product_name)

                # unload bands

                # plot the graphs specified in graph_bands
                graph_basename = saver.output_folder.joinpath(product_name).joinpath(saver.base_name + product_name)\
                    .as_posix()
                DWutils.plot_graphs(bands, self.bands_graphs, matrice_cluster,
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

# check if the system was called from the main flux
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The products input folder. Required.", required=True, type=str)
    parser.add_argument("-o", "--out", help="Output directory. Required.", required=True, type=str)
    parser.add_argument("-s", "--shp", help="SHP file. Required.", type=str)
    parser.add_argument("-p", "--product", help='The product to be processed (S2_Theia, Landsat, etc.)',
                        default='S2_THEIA', type=str)
    parser.add_argument('-g', '--off_graphs', help='Turns off the scatter plot graphs', action='store_true')

    # product type (theia, sen2cor, landsat, etc.)
    # optional shape file
    # generate graphics (boolean)
    # name of config file with the bands-list for detecting, saving graphics, etc. If not specified, use default name
    #   if clip MIR or not, number of pixels to plot in graph, number of clusters, max pixels to process, etc.
    # masks to be considered?!?!?

    args = parser.parse_args()

    water_detect = WaterDetect(input_folder=args.input, output_folder=args.out, shape_file=args.shp,
                               product=args.product)
    water_detect.run()

    # print(water_detect.input_folder)

    # list of images folders
    # loop through it
    # process each image

    # DetectWater wrapper class
    #     - O wrapper deve criar um loader
    #     - O loader vai ter X imagens e para cada X imagens, Y batches
    #     - fazer um loop enquanto o loader != empty
    #       - DWImage.process (loader.pop())

    # DWLoader (proxy entre o produto no disco e as imagens no formato necessário)
    #   - abre o produto
    #   - .load()
    #   - .bands (dictionary of bands)
    #   - .masks (dictionary of masks)
    #   - if Batch o DWImage vai pegar com DWLoader.Pop() e processar
    #       - na verdade o Pop faz o load da fila
    #   -
    # DWImage
    #   -