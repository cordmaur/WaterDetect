# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:56:21 2020

@author: MH
"""
#import fiona
#import pprint

import os 
import glob
#import shutil
import numpy as np
from osgeo import gdal
#from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path




class DWSerie:

    def __init__(self, input_folder, parameters, shape_file):#,  ref_band):
        self.input_folder = input_folder
        self.parameters = parameters
        self.shape_file = shape_file
#-----------------------MAURICIO
    @staticmethod
    def check_path(path_str, is_dir=False):
        """
        Check if the path/file exists and returns a Path variable with it
        :param path_str: path string to test
        :param is_dir: whether if it is a directory or a file
        :return: Path type variable
        """

        if path_str is None:
            return None

        path = Path(path_str)

        if is_dir:
            if not path.is_dir():
                raise OSError('The specified folder {} does not exist'.format(path_str))
        else:
            if not path.exists():
                raise OSError('The specified file {} does not exist'.format(path_str))

        # print(('Folder' if is_dir else 'File') + ' {} verified.'.format(path_str))
        return path

    def get_directories(self, input_folder):
        """
        Return a list of directories in the folder given. These folders are the repository for satellite products
        
        :param input_folder: folder that stores the images
        :return: list of images (i.e. directories)
        """
        return [str(i) for i in input_folder.iterdir() if i.is_dir()]
#-----------------------------------------------------------------------------------------------
    def get_images(self, files):
        images = []
        for file in files:
            file = Path(file)
            images.append([str(i) for i in file.iterdir() if i.is_dir()])
        return images

    def clip_param(self, shape_file, parameters, files):
        """Function to clip a tif from a shapefile, 
        resulting in creating a new tiff with the layer extent of the shapefile
        
        :param shape_file: polygon shapefile that will be used to clip the tiff file
        :param parameters: list of parameters we want to clip
        :param files: list of directories
        :return: list of clipped files to process next
        """

        opt = gdal.WarpOptions(cutlineDSName=self.shape_file, cropToCutline=True,
                                srcNodata=-9999, dstNodata=-9999, outputType=gdal.GDT_Float32)
        newfiles = []
        for file in files:
            os.makedirs(file+'\\'+"Clip", exist_ok=True)
            newfiles.append(file+'\\'+"Clip")
            for param in parameters:
                if self.check_path(file+'\\'+param+".tif"):
                    dest_name = file+'\\'+"Clip"+'\\'+param+ "_clipped.tif"
                    gdal.Warp(destNameOrDestDS=dest_name, srcDSOrSrcDSTab=file+'\\'+param+'.tif', options=opt)
                else:
                    print("Parameters files do not exist")
                    os.remove(file+'\\'+"Clip")
        return newfiles


    def extract_mean(self, filename, suffix):
        """
        Return the positive mean value of a raster
        
        :param filename: name of the raster file from which we want the mean value
        :param suffix: param for which we want the mean value
    	:return: mean value
        """
        # Ouverture dataset
        ds=gdal.Open(filename+'\\'+suffix)
        # RasterBand
        band = ds.GetRasterBand(1)
        # Array
        rasterData = band.ReadAsArray()
        # mean value among the positive values of the array
        mean = np.mean(rasterData[rasterData>0])
        return mean

    def extract_date_csv(self, filename):
        """
        Return the first and last date corresponding to the batch of input image
        
        :param filename: name of the raster file from which we extract the date
    	:return: first and last date of the batch
        """
        # first_date = filename[0].split('\\')[-1].split('_')[1].split('-')[0]
        # last_date = filename[-1].split('\\')[-1].split('_')[1].split('-')[0]
        first_date = filename[0].split('\\')[-1]
        last_date = filename[-1].split('\\')[-1]
        return [first_date, last_date]
            
    
    def extract_date(self, filename):
        """
        Return the date corresponding to the input image
        
        :param filename: name of the raster file from which we extract the date
    	:return: date (format 2020-01-01)
        """
        fullsp = filename.split('\\')
        for name in fullsp:
            if name.find('SENTINEL') == 0:
                indice = fullsp.index(name)
        sp = str(fullsp[indice]).split('_')
        date = sp[1].split('-')[0]
        date_csv = date[:4]+"-"+date[4]+date[5]+"-"+date[6:]
        return date_csv
    
    def extract_tilename(self, filename):
        """
        Return the tilename corresponding to the input image
        
        :param filename: name of the raster file from which we extract the tilename
    	:return: tilename (example T31TFJ)
        """
        namelist = filename.split('\\')
        for name in namelist:
            if name.find('T') == 0:
                indice = namelist.index(name)
        tilename = namelist[indice]
        return tilename
    
    def create_csv(self, path):
        with open(str(path)+'\\'+"filecsv.csv", "a") as f:
            f.write("Date"+ "," +self.parameters[0] + "," +self.parameters[1] + "\n")
            f.close()
    
    def fill_csv(self, paths, datelist, output):
        """
        Create a csv file with Date and 2 parameters
        
        :param paths: list of the images wanted to be processed
        :param datelist:
        :param output:
        :return: name of the csv file created
        """
        pathlist = []
        output = str(output)

        for path in paths:
            # path: D:\\S2\\Output_Wat\\France\\T31TFJ\\2019-07\\SENTINEL2A_20190722-104855-308_L2A_T31TFJ_C_V2-2
            if len(path) > 1:
                for file in path:
                    pathlist.append(file + '\\' + "Clip")
                    files = os.listdir(file + '\\' + "Clip")
            else:
                pathlist.append(path[0] + '\\' + "Clip")
                files = os.listdir(path[0] + '\\' + "Clip")
            # pathlist.append(path)
            # files = os.listdir(path)


        with open(output+'\\'+"filecsv.csv", "a") as f:
            for path in pathlist:
                param = []
                date = self.extract_date(path)
                datelist.append(date)
                tilename = self.extract_tilename(path)
                for suffixe in files:
                    # Extract mean value
                    param.append(self.extract_mean(path, suffixe))
                f.write(date + "," +str(param[0]) + "," +str(param[1])+"\n") # line in a txt

        # Getting first and last date of the files processed
        namecsv = output+"\\"+"S2-"+tilename+"_"+datelist[0]+"_"+datelist[-1]+".csv"

        # deletes csv file if already exists
        try:
            os.remove(namecsv)
        except:
            print("File not existing before ", namecsv)
        os.rename(output+'\\'+"filecsv.csv", namecsv)
        print("File created", namecsv)
        return namecsv
        
    def save_graph_csv(self, output, filecsv):
        """
        Save a time series graph from a csv file
        :output: output folder
        :param filecsv: csv file
        """
        output = str(output)
        data = pd.read_csv(filecsv, header=0, parse_dates=True, keep_date_col=True, squeeze=True)
        
        # Initialize lists to be plotted
        x = []
        y = []
        
        # Initializing Date column
        Date = data["Date"]
        x = list(Date)
        # Get column names
        column_names = data.columns.values.tolist()
        column_names.pop(0) # removing date column
       
        # Associating column name with data
        for i in range(len(column_names)):
            column_names[i] = data[column_names[i]]
            y.append(list(column_names[i]))
        
        # Saving one figure per parameter
        for i in range(len(y)):
            plt.figure()
            plt.plot(x,y[i], marker='o')
            plt.grid(b=True, which='major', axis='both')
            plt.xlabel('Date')
            plt.ylabel(data.columns.values[i+1])
            #plt.savefig(output+'\\'+data.columns.values[i+1]+'.png')
            plt.savefig(output+'\\'+data.columns.values[i+1]+'.pdf')
            plt.close()
            
            
    def old_plot_csv(self, filecsv, parameters):
        """
        Plot data from a csv file
        
        :param filecsv: csv file
        :param parameters: list of parameters wanted to be represented
        """
        series = pd.read_csv(filecsv, header=0, parse_dates=True, keep_date_col=True, squeeze=True)
        #df['col'] = pd.to_datetime(df['col'])
        series['Date'] = pd.to_datetime(series['Date'])
        series.set_index('Date', inplace=True)

        series['Date'] = pd.to_datetime(series['Date'])
        
        #Add list of suffixes
        series.plot(x=series.columns[0], y=series.columns[1], kind = 'scatter')
        series[parameters[0]].plot(color='g', kind = 'scatter', lw=1.3) 
        series[parameters[1]].plot(color='r',lw=1.3)
        
        series.plot(x=series['Date'],  y=series.columns)#style='k.'
        series.plot(x=series.columns[0], y=series.columns)
        series.plot(kind="scatter")
        
        plt.show()
        
        
    def run(self):
        # Listing images in the input directory
        path = self.check_path(self.input_folder)
        files = self.get_directories(path)
        self.create_csv(path)
        datelist = self.extract_date_csv(files)
        #print(datelist)
        image = self.get_images(files)
        for im in image:
            clipped_files = self.clip_param(self.shape_file, self.parameters, im)
        # Creating a csv file with 3 columns: Date and 2 parameters
        namecsv = self.fill_csv(image, datelist, self.input_folder)
        # Plotting the results
        self.save_graph_csv(self.input_folder,namecsv)

        #df_mod_srtd['Datetime'] = pd.to_datetime(df_mod_srtd['Date'], errors='coerce')