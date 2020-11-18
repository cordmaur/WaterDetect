# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author:
Corresponding Author :
Project:
Created on:
"""

import DWWaterDetect
import argparse

# check if this file has been called as script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The products input folder. Required.", required=True, type=str)
    parser.add_argument("-o", "--out", help="Output directory. Required.", required=True, type=str)
    parser.add_argument("-s", "--shp", help="SHP file. Optional.", type=str)
    parser.add_argument("-p", "--product", help='The product to be processed (S2_THEIA, LANDSAR or S2_L1C)',
                        default='S2_THEIA', type=str)
    parser.add_argument('-c', '--config', help='Configuration .ini file. If not specified WaterDetect.ini '
                                               'is used as default', type=str)

    # product type (theia, sen2cor, landsat, etc.)
    # optional shape file
    # generate graphics (boolean)
    # name of config file with the bands-list for detecting, saving graphics, etc. If not specified, use default name
    #   if clip MIR or not, number of pixels to plot in graph, number of clusters, max pixels to process, etc.
    # name of the configuration .ini file (optional, default is WaterDetect.ini in the same folder

    args = parser.parse_args()

    water_detect = DWWaterDetect.DWWaterDetect(input_folder=args.input, output_folder=args.out, shape_file=args.shp,
                                               product=args.product, config_file=args.config)
    water_detect.run()
