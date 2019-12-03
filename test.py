from matplotlib import colors
from osgeo import gdal
from osgeo import osr
from pathlib import Path
from PyPDF2 import PdfFileMerger
from scipy.spatial import ConvexHull, Delaunay
from shutil import copy
from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from unittest import TestCase
import argparse
import ast
import configparser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pysptools.abundance_maps as abundance_maps
import sys
matplotlib.use('Agg')

if __name__ == '__main__':
    print('hello world!')