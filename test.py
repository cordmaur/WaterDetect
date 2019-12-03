from matplotlib import colors #ok
from osgeo import gdal
from osgeo import osr
from pathlib import Path #ok
from PyPDF2 import PdfFileMerger
from scipy.spatial import ConvexHull, Delaunay
from shutil import copy
from sklearn import cluster #ok
from sklearn import metrics #ok
from sklearn import preprocessing #ok
from sklearn.cross_validation import cross_val_score #ok
from sklearn.decomposition import PCA #ok
from sklearn.grid_search import GridSearchCV #ok
from sklearn.mixture import GMM #ok
from sklearn.model_selection import train_test_split #ok
from sklearn.naive_bayes import GaussianNB #ok
from sklearn.neural_network import MLPClassifier #ok
from sklearn.svm import LinearSVC #ok
from unittest import TestCase #ok
import argparse #ok
import ast #ok
import configparser #ok
import matplotlib #ok
import matplotlib.pyplot as plt #ok
import numpy as np #ok
import os #ok
import pysptools.abundance_maps as abundance_maps #ok
import sys #ok
matplotlib.use('Agg')

if __name__ == '__main__':
    print('hello world!')