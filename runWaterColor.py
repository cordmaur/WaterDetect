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

from pathlib import Path
import CalculateStatistics2


class WaterDetect:
    def __init__(self, input_folder, output_folder, shape_file, product):

        self.input_folder = self.check_path(input_folder, is_dir=True)
        self.output_folder = self.check_path(output_folder, is_dir=True)
        self.shape_file = self.check_path(shape_file, is_dir=False)

        self.images = self.load_input_images(self.input_folder)

        self.product = product
        return

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
    def load_input_images(input_folder):
        """
        Return a list of directories in input_folder. These folders are the repository for satellite products
        :param input_folder: folder that stores the images
        :return: list of images (i.e. directories)
        """
        return [i for i in input_folder.iterdir() if i.is_dir()]

    def load_product_bands(self, image_folder):

        print('Retrieving bands for image: ' + image_folder.as_posix())
        if self.product == 'S2_THEIA':
            # get flat reflectance bands in a list
            bands = [file for file in image_folder.iterdir() if
                     file .suffix == '.tif' and 'FRE' in file.stem and 'V1' in file.stem]
            for b in bands:
                print(b.stem)
        else:
            bands = None

        return bands

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

    def run(self):

        for image in self.images:

            bands = self.load_product_bands(image)
            masks = self.load_mask_bands(image)

            # if there are bands loaded call the water detection algorithm
            if bands:
                CalculateStatistics2.Treat_files(bands, masks, self.product, self.output_folder, self.shape_file)

        return

# check if the system was called from the main flux
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The products input folder. Required.", required=True, type=str)
    parser.add_argument("-o", "--out", help="Output directory. Required.", required=True, type=str)
    parser.add_argument("-s", "--shp", help="SHP file. Required.", type=str)
    parser.add_argument("-p", "--product", help='The product to be processed (S2_Theia, Landsat, etc.)',
                        default='S2_Theia', type=str)
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

    print(water_detect.input_folder)

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