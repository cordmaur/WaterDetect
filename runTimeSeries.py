# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:56:44 2020

@author: MH
"""

import argparse

import DWSerie


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="The products in the output folder. Required.", required=True, type=str)
    parser.add_argument("-pa", "--parameters", nargs='+', help="Name of the parameters for the time series in the list: spm-get,turb-dogliotti,chl-lins,aCDOM-brezonik. Required.", required=True)
    parser.add_argument("-s", "--shp", help="SHP file. Optional.", type=str)
    
    args = parser.parse_args()

    time_series = DWSerie.DWSerie(input_folder=args.output, parameters = args.parameters, shape_file=args.shp)
                                               #product=args.product, config_file=args.config)
    time_series.run()