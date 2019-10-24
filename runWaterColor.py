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

import DWWaterDetect
import argparse


# check if the system was called from the main flux
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The products input folder. Required.", required=True, type=str)
    parser.add_argument("-o", "--out", help="Output directory. Required.", required=True, type=str)
    parser.add_argument("-s", "--shp", help="SHP file. Optional.", type=str)
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

    water_detect = DWWaterDetect.DWWaterDetect(input_folder=args.input, output_folder=args.out, shape_file=args.shp,
                                               product=args.product)
    water_detect.run()


# -s ../../source/Shp/Area_Chad.shp -p S2_THEIA