import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import pysptools.abundance_maps as abundance_maps
from matplotlib import colors
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

from scipy.spatial import ConvexHull, Delaunay
from shutil import copy

dicS2BandNames     = {'Blue': 'B2', 'Green': 'B3', 'Red': 'B4', 'Mir': 'B11', 'Mir2': 'B12', 'Nir': 'B8', 'Nir2': 'B8A'}
dicL8USGSBandNames = {'Green': 'B3', 'Red': 'B4', 'Mir': 'B6', 'Nir': 'B5'}
dicOtherBandNames  = {'Green': 'band3', 'Red': 'band4', 'Mir': 'band6', 'Nir': 'band5'}

dicDownloadProducts = {'S2': dicS2BandNames,
                       'L8USGS': dicL8USGSBandNames,
                       'Other': dicOtherBandNames}


def Treat_files(band_clip, mask_clip, Download, Temp_name, shp_name):

    # initialize the variables
    max_invalid_pixels = 0.8    # The maximum percentage of invalid (masked) pixels to continue
    min_mndwi = 0.0             # mndwi threshold
    clustering = 'aglomerative' # aglomerative, kmeans, gauss_mixture
    classifier = 'naive_bayes'  # naive_bays, MLP, Hull, SVM
    clip_mndwi = 0.05           # None or mndwi value to clip false positives
    ref_band   = 'Red'

    bands_keys = [['Mir2', 'mndwi']]
    # bands_keys = [['Mir2', 'mndwi'],
    #               ['mndwi', 'ndwi']]


    # bands_keys = [['Red', 'Nir2'],
    #               ['Red', 'Mir'],
    #               ['Red', 'Nir2', 'Mir'],
    #               ['Red', 'Nir2', 'mndwi'],
    #               ['Red', 'Mir', 'mndwi'],
    #               ['Red', 'Nir2', 'Mir', 'mndwi'],
    #               ['Red', 'mndwi', 'ndwi'],
    #               ['Nir2', 'Mir'],
    #               ['Mir', 'mndwi'],
    #               ['Nir2', 'Mir', 'mndwi'],
    #               # ['PCA', '3'],
    #               ['PCA', '2'],
    #               ['Red', 'mndwi']]

    graph_bands = [['Mir2', 'mndwi']]
    # graph_bands = [['Red', 'Nir2'],
    #                ['Red', 'Mir2'],
    #                ['Red', 'mndwi'],
    #                ['Mir2', 'mndwi'],
    #                ['Nir2', 'mndwi'],
    #                ['Nir2', 'Mir2'],
    #                ['ndwi', 'mndwi']]

    # create output folder with given path and shape name (area to compute water masks)
    Temp_name = createOutputDirectory(Temp_name, shp_name)

    # Open the gdal bands according to the download product
    geo_bands = loadBandsByDownload(Download, band_clip)

    # read GeoTransform and Projection to write new files
    geotransform = geo_bands[ref_band].GetGeoTransform()
    projection   = geo_bands[ref_band].GetProjection()

    # create output composites (VRTs) with RGB clipped bands
    createComposite(geo_bands, os.path.split(Temp_name)[0])

    # Reshape RasterImages bands and masks (first image is the baseline)
    # reshapeImages(list(bands.values()))

    bands = loadRasterImages(geo_bands, ref_band=ref_band)
    x_size = list(bands.values())[0].shape[1]
    y_size = list(bands.values())[0].shape[0]

    # Load the masks (theia S2)
    # All these masks have true in the INVALID pixels
    # cloud_mask = getTheiaCloudMask(mask_clip[0][0], clouds=True, shadows=True, x_size=x_size, y_size=y_size)
    # nodata_mask = getTheiaEDGMask(mask_clip[0][1], x_size=x_size, y_size=y_size)
    # mg2_mask = getTheiaMG2Mask(mask_clip[0][2], x_size=x_size, y_size=y_size)
    # sat_mask = getTheiaSatMask(mask_clip[0][3], x_size=x_size, y_size=y_size)

    # Create a invalid combined mask excluding the no data pixels in ALL BANDS
    # invalid_mask = cloud_mask | nodata_mask | mg2_mask | sat_mask
    invalid_mask = False

    for key in bands.keys():
        invalid_mask |= (bands[key] == -9999)  # | (bands[key].ReadAsArray() < 0)

    # Test if there is enough valid pixels in the clipped images
    if (np.count_nonzero(invalid_mask) / invalid_mask.size) > max_invalid_pixels:
        print('Not enough valid pixels in the image area')
        return

    # calc the MNDWI water index and its mask according to min_mndwi threshold
    mndwi, water_mndwi_mask = calcMndwi(bands, invalid_mask, min_mndwi)
    # add the mndwi as a band in the bands dictionary
    # careful because ohter bands are gdal bands and mndwi is an array
    bands.update({'mndwi': mndwi.filled()})

    ndwi = calcNdwi(bands, invalid_mask)
    bands.update({'ndwi': ndwi.filled()})

    # calc NDVI vegetationi index
    ndvi = calcNDVI(bands, invalid_mask)
    bands.update({'ndvi': ndvi.filled()})

    # update the invalid pixels mask with the result from MNDWI (infinite pixels)
    invalid_mask = invalid_mask | mndwi.mask | ndvi.mask | ndwi.mask

    # save mndwi image and mask
    array2raster(Temp_name + '_MNDWI.tif', mndwi.filled(), geotransform, projection, gdal.GDT_Float64, -9999)
    array2raster(Temp_name + '_MNDWI_mask.tif', water_mndwi_mask, geotransform, projection, gdal.GDT_Byte, 0)

    # runs the detect water algorithm and plot graphs and saves the TIFs
    processDetectWater(bands, bands_keys, graph_bands, invalid_mask, Temp_name, clustering=clustering,
                       classifier=classifier, min_clusters=2, max_clusters=4, clip_mndwi=clip_mndwi,
                       geotransform=geotransform, projection=projection)

    # sys.exit()
    return


def processDetectWater(geo_bands, bands_keys, graph_bands, invalid_mask, base_filename, geotransform, projection,
                       clustering='aglomerative', classifier='naive_bayes', min_clusters=3, max_clusters=6,
                       clip_mndwi=None):

    # get the base directory name
    base_folder = os.path.split(base_filename)[0]

    # if bands_keys is not a list of lists, transform it
    if type(bands_keys[0]) == str:
        bands_keys = [bands_keys]

    # loop through the bands combinations to make the clusters
    for bands_combination in bands_keys:

        # prepare the base product name based on algorithm and bands, to create the directory
        product_name = createProductName(bands_combination, clustering,
                                         clip_mndwi=clip_mndwi)  # Create the product directory
        dir_name = os.path.join(base_folder, product_name)
        os.makedirs(dir_name, exist_ok=True)

        # Create the full path filename with the tile name
        filename = os.path.join(dir_name, os.path.split(base_filename)[-1] + product_name)

        # detect the water pixels and stores a matrix with the clusters
        # water=1 Vegetation=2
        try:
            matrice_cluster = detectWater(geo_bands, bands_combination, invalid_mask,
                                          clustering=clustering,
                                          classifier=classifier,
                                          min_clusters=min_clusters,
                                          max_clusters=max_clusters,
                                          clip_mndwi=clip_mndwi)

            # plot the graphs specified in graphbands
            if graph_bands:
                plotGraphs(geo_bands, graph_bands, matrice_cluster, filename, invalid_mask, 1000)

            # plotPCA(geo_bands, matrice_cluster, filename, invalid_mask, 1000)

            # export the clustered matrix as a raster
            array2raster(filename + '_Clusters.tif', matrice_cluster, geotransform, projection, gdal.GDT_Byte, 0)

            array2raster(filename + '_WaterMask.tif', matrice_cluster == 1, geotransform, projection, gdal.GDT_Byte, 0)

        except Exception as err:
            print("############# ERROR ################")
            print(err)

    return


def detectWater(bands, bands_keys, invalid_mask, clustering='kmeans',
                classifier='naive_bayes', min_clusters=3, max_clusters=6, clip_mndwi=None):
    """
    :param bands: list of all bands as numpy arrays available as a dictionary
    :param bands_keys: name of the bands to use in the water detection. This function assumes RED as the first band and Infrared
    as the second band. All other bands will be used as aditional bands in clustering
    :param invalid_mask: mask with the invalid pixels to be discarded
    :param clustering: Clustering algorithm to use (kmeans, aglomerative, gaussian mixture, etc.)
    :param classifier: supervised classifier to generalize the clustering result
    :param min_clusters: minimum number of clusters
    :param max_clusters: maximum number of clusters
    :return: cluster mask indicating water (1), vegetation (2) and other clusters as 3, 4, ...
    """

    # Transform the rasters in a matrix where each band is a column
    original_data = bandsToColumns(bands, invalid_mask)

    # two line vectors indicating the indexes (line, column) of valid pixels
    ind_data = np.where(~invalid_mask)

    # if algorithm is not kmeans, split data for a smaller set (for performance purposes)
    if clustering == 'kmeans':
        original_train_data = original_data
    else:
        # original train data keeps all the bands
        original_train_data, _ = getTrainTestDataset(original_data, 0.1, min_train_size=1000, max_train_size=10000)

    # if the bands combination is PCA, decomposes the bands in the Principal Components
    if bands_keys[0] == 'PCA':
        pca = PCA(n_components=int(bands_keys[1])).fit(splitDataByBands(original_data, bands, ['Nir2', 'Mir', 'mndwi', 'Red']))
        train_data = pca.transform(splitDataByBands(original_train_data, bands, ['Nir2', 'Mir', 'mndwi', 'Red']))
        data = pca.transform(splitDataByBands(original_data, bands, ['Nir2', 'Mir', 'mndwi', 'Red']))
    else:
        train_data = splitDataByBands(original_train_data, bands, bands_keys)
        data = splitDataByBands(original_data, bands, bands_keys)

    # find the best clustering solution (k = number of clusters)
    best_k = findBestK(train_data, score_index='calinsk', min_k=min_clusters, max_k=max_clusters, train_size=0.1)

    # apply the clusterization algorithm and return labels and train dataset
    clusters_labels = applyCluster(train_data, algorithm=clustering, k=best_k, train_size=0.1)

    # get the vegetation cluster
    clusters_params = calcClustersParams(original_train_data, clusters_labels)

    # indentify the vegetation cluster using the maximum NDVI
    veg_cluster = detectVegetationCluster(clusters_params, bands)

    # detect the water cluster
    water_cluster = detectWaterCluster(original_train_data, bands, ['Red', 'Mir'],
                                       clusters_params, veg_cluster, method='minmir2', train_size=0.1)

    # if we are dealing with aglomerative cluster or other diff from kmeans, we have only a sample of labels
    # we need to recreate labels for all the points using supervised classification
    if clustering != 'kmeans':
        clusters_labels = supervisedClassification(data, train_data, clusters_labels, classifier)
        # clusters_labels = supervisedClassification(original_data[:, 0:6], original_train_data[:, 0:6], clusters_labels, classifier)

    # after obtaining the final labels, if clip MIR is not None, clip MIR above threshold
    if clip_mndwi:
        clusters_labels[(clusters_labels==water_cluster['clusterid']) & (bands['Mir2'][~invalid_mask] > clip_mndwi)] = -1
        # clusters_labels[(clusters_labels == water_cluster['clusterid']) & (geo_bands['mndwi'][~invalid_mask] < clip_mndwi)] = -1

    # create an cluster array based on the cluster result (water will be value 1)
    return createMatriceCluster(list(bands.values())[0], ind_data, clusters_labels, water_cluster, veg_cluster)


############################################################################
# Raster & Array Utility functions
# -------------------------------------------------------------------------#
def splitDataByBands(data, geo_bands, selected_keys):
    bands_index = []
    bands_keys = list(sorted(geo_bands.keys()))

    for key in selected_keys:
        bands_index.append(bands_keys.index(key))

    return data[:, bands_index]


def bandsToColumns(geo_bands, invalid_mask):
    # load all the bands in the dictionary as numpy arrays columns
    # the bands will appear in sorted order
    valid_data_list = []
    bands_list = []

    for key in sorted(geo_bands.keys()):
        band_array = geo_bands[key]

        bands_list.append(key)
        valid_data_list.append(band_array[~invalid_mask])

    # prepare the multidimensional data array (bands as columns)
    data = np.c_[valid_data_list].transpose()

    return data


def array2raster(newRasterfn, array, geo_transform, projection, datatype, nodatavalue=0):

    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(geo_transform)
    outRaster.SetProjection(projection)
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(nodatavalue)
    outband.WriteArray(array)
    outband.FlushCache()
    print('Saving image: ' + newRasterfn)
    return


def reshapeImages(images_list):
    # one way of doing this is using gdal.Translate('new file name.tif', dataset to be reshaped, gdalTranslateOptions)
    # t_options = gdal.TranslateOptions(width=5490, height=5490)
    # resampled=gdal.Translate('../-599_L2A_T22KGA_D_V1-9_FRE_B2_20m.tif', ds, options=t_options)

    for i in range(len(images_list)):
        if i != 0:
            band_src = images_list[i]
            band_tgt = images_list[i-1]
            if (band_src.RasterXSize != band_tgt.RasterXSize) | (band_src.RasterYSize != band_tgt.RasterYSize):
                raise OSError("Raster images with different sizes")
            #TODO: Reshape must be done in matrix level, not GDAL
    return


############################################################################
# Functions to manage the loading of the clipped THEIA BANDS
# -------------------------------------------------------------------------#
def getImageByBandId(images_list, band):
    """
    Get the image in the list corresponding to the informed Band.
    Return the image opened with GDAL as a RasterImage object
    If cant find the band return None
    If is more than 1 image, raise exception
    """
    band = '_' + band + '.'
    image_band = list(filter(lambda x: band in os.path.split(x)[-1], images_list))

    if len(image_band) == 0:
        return None, None
    elif len(image_band) > 1:
        raise OSError('More than one band {} in image list'.format(band))

    gdal_ds = gdal.Open(image_band[0].as_posix())

    if not gdal_ds:
        raise OSError("Couldn't open band file {}".format(image_band[0]))

    return gdal_ds


def selectBandNames(band_names, band_ids):
    selected_band_names = {}
    for band in band_names:
        if band_names[band] in band_ids:
            selected_band_names.update({band: band_names[band]})

    return selected_band_names


def loadImagesByDict(images_list, bands_dict):
    """
    Load a bands list, given a image_list and a dictionary of Keys(BandName) and identifiers to parse the filename
    ex. {'Green':'B3', 'Red':'B4'...}
    The result, will be a dictionary with Keys(BandName) and RasterImages as values
    """
    bands = {}
    for band_key in bands_dict:
        print('Loading band: ' + band_key)
        gdal_img = getImageByBandId(images_list, bands_dict[band_key])
        bands.update({band_key: gdal_img})

    return bands


def loadBandsByDownload(download, images_list):

    band_names = getBandsNames(download)

    return loadImagesByDict(images_list, band_names)


def getBandsNames(download):
    if download in ["L8_THEIA", "S5_THEIA"]:
        print('not yet implemented')
        sys.exit()

    else:
        if 'L8_USGS' in download:
            band_names = dicL8USGSBandNames
        elif download in ["S2_PEPS", "S2_S2COR", "S2_THEIA", "S2_L2H"]:
            band_names = dicS2BandNames
        else:
            band_names = dicOtherBandNames
    return band_names


def loadRasterImages(geo_bands, ref_band):

    raster_bands = {}

    # getting the reference raster size as RED band
    ref_band = geo_bands[ref_band]
    x_size, y_size = int(ref_band.RasterXSize), int(ref_band.RasterYSize)

    # resizing bands as Red band
    for band_key in geo_bands.keys():
        gdal_img = geo_bands[band_key]
        raster_bands.update({band_key: gdal_img.ReadAsArray(buf_xsize=x_size, buf_ysize=y_size)/10000})
        del gdal_img

    return raster_bands


############################################################################
# Functions to manage THEIA MASKS
# -------------------------------------------------------------------------#
def getTheiaCloudMask(img_path, clouds=True, shadows=True, x_size=None, y_size=None):
    mask_img = gdal.Open(img_path.as_posix())
    if not mask_img:
        raise OSError("Can't open file {}".format(img_path.as_posix()))

    if clouds & shadows:
        return mask_img.ReadAsArray(xsize=x_size, ysize=y_size) != 0
    else:
        return mask_img.ReadAsArray(xsize=x_size, ysize=y_size) != 0


def getTheiaEDGMask(img_path, x_size=None, y_size=None):
    mask_img = gdal.Open(img_path.as_posix())
    if not mask_img:
        raise OSError("Can't open file {}".format(img_path.as_posix()))

    return mask_img.ReadAsArray(xsize=x_size, ysize=y_size) != 0


def getTheiaMG2Mask(img_path, x_size=None, y_size=None):
    mask_img = gdal.Open(img_path.as_posix())
    if not mask_img:
        raise OSError("Can't open file {}".format(img_path.as_posix()))

    array = mask_img.ReadAsArray(xsize=x_size, ysize=y_size)
    return (array != 0) & (array != 1) & (array != 4)


def getTheiaSatMask(img_path, x_size=None, y_size=None):
    mask_img = gdal.Open(img_path.as_posix())
    if not mask_img:
        print("Can't open file " + img_path.as_posix())
        raise OSError("Can't open file {}".format(img_path.as_posix()))

    return mask_img.ReadAsArray(xsize=x_size, ysize=y_size) != 0


############################################################################
# Machine Learning functions
# -------------------------------------------------------------------------#
def supervisedClassification(data, train_data, clusters_labels, algorithm):
    if algorithm == 'aglomerative':
        clusters_labels = applyConvexHull(data, clusters_labels, train_data)
    elif algorithm == 'naive_bayes':
        clusters_labels = applyNaiveBayes(data, clusters_labels, train_data)
    elif algorithm == 'MLP':
        clusters_labels = applyMLP(data, clusters_labels, train_data)
    elif algorithm == 'SVM':
        clusters_labels = applySVM(data, clusters_labels, train_data)

    return clusters_labels


def applySVM(data, clusters_labels, clusters_data):

    # train a SVM classifier with the data and labels provided
    # before it, test a grid to check the best C parameter
    grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0, 50, 100, 100000]})

    grid.fit(clusters_data, clusters_labels)

    print('Applying clusters Suport Vector Machine classifier')
    print('Cross_val_score:{}'.format(grid.best_score_))

    model = grid.best_estimator_
    model.fit(clusters_data, clusters_labels)

    # return the new predicted labels for the whole dataset
    return model.predict(data)


def applyNaiveBayes(data, clusters_labels, clusters_data):
    # train a NB classifier with the data and labels provided
    model = GaussianNB()

    print('Applying clusters based naive bayes classifier')
    # print('Cross_val_score:{}'.format(cross_val_score(model, clusters_data, clusters_labels)))

    model.fit(clusters_data, clusters_labels)

    # return the new predicted labels for the whole dataset
    return model.predict(data)


def applyMLP(data, clusters_labels, clusters_data):

    clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(8, 8), random_state=1)

    print('Applying clusters based Multilayer Perceptron classifier')
    print('Cross_val_score:{}'.format(cross_val_score(clf, clusters_data, clusters_labels)))

    clf.fit(clusters_data, clusters_labels)

    return clf.predict(data)


def applyConvexHull(data, clusters_labels, clusters_data):
    print('Applying clusters based on convex hull of trained data')

    # create the new labels vector and initializes it with -1
    new_labels = np.zeros(data.shape[0]) - 1

    # for each cluster, creaete a convex hull and select the points inside it
    for label_i in np.unique(clusters_labels):
        # for performance purposes, first get the outter vertices of the hull
        # hull = ConvexHull(clusters_data[clusters_labels == label_i, :])
        # outter_vertices = clusters_data[hull.vertices, :]

        # after that, we get a triangulation of the outter vertices
        # hull = Delaunay(outter_vertices)
        hull = Delaunay(clusters_data[clusters_labels == label_i, :])
        label_i_index = hull.find_simplex(data, tol=0.5) >= 0
        new_labels[label_i_index == True] = label_i

    return new_labels


############################################################################
# Clustering utility functions
# -------------------------------------------------------------------------#
def calcClustersParams(data, clusters_labels):

    clusters_params = []
    for label_i in np.unique(clusters_labels):
        # first slice the values in the indexed cluster
        cluster_i = data[clusters_labels == label_i, :]

        cluster_param = {'clusterid': label_i}
        cluster_param.update({'mean': np.mean(cluster_i, 0)})
        cluster_param.update({'variance': np.var(cluster_i, 0)})
        cluster_param.update({'stdev': np.std(cluster_i, 0)})
        cluster_param.update({'diffb2b1': cluster_param['mean'][1] - cluster_param['mean'][0]})
        clusters_params.append(cluster_param)

    return clusters_params


def detectWaterCluster(data, geo_bands, band_keys, clusters_params, veg_cluster, method='maxdistance', train_size=0.4):

    print('Detecting the water cluster by: ' + method)

    if method == 'maxdistance':
        mindist = 0
        for cluster_ in clusters_params:
            if cluster_ != veg_cluster:
                dist = np.linalg.norm(veg_cluster['mean'] - cluster_['mean'])
                print(cluster_)
                print('Dist to vegetation member {}'.format(dist))
                if dist > mindist:
                    mindist = dist
                    water_cluster = cluster_

    elif method == 'maxmndwi':
        water_cluster = detectCluster(clusters_params, sorted(geo_bands.keys()), 'value', 'max', 'mndwi')

    elif method == 'minmir2':
        water_cluster = detectCluster(clusters_params, sorted(geo_bands.keys()), 'value', 'min', 'Mir2')

    elif method == 'minerror':
        # get a random sample of the data to extract abundances and test error
        # train_data, test_data = getTrainTestDataset(data, train_size, min_train_size=1000, max_train_size=5000)


        # modified version: instead of calculating mean error for all points, calcule error only
        # for the centroids
        train_data = []
        for cluster_ in clusters_params:
            train_data.append(cluster_['mean'])

        train_data = splitDataByBands(np.array(train_data), geo_bands, band_keys)

        # NNLS receives a (n x m x p)but we dont have the bidimensional raster
        # p is the number of multispectral bands
        # we will include an axis as the second dimension, to form single column raster
        # train_data[:,np.newaxis,:]

        # initialize the error variable with the maximum number possible
        min_error = sys.float_info.max

        for cluster_ in clusters_params:
            if cluster_ != veg_cluster:
                # assume the current cluster is the water cluster
                endmembers = np.c_[veg_cluster['mean'], cluster_['mean']].transpose()
                endmembers = splitDataByBands(endmembers, geo_bands, band_keys)

                # calc the the abundance in termos of water and vegetation for the pixels
                nnls = abundance_maps.NNLS()
                amap = nnls.map(M=train_data[:, np.newaxis, :], U=endmembers)

                # return data to the 2 dimensions original
                amap = amap[:, 0, :]

                # if we multiply the abundances by the endmembers we should get the original(predicted) values
                data_predict = amap.dot(endmembers)

                # the error is the sum of squared difference between original data and predicted
                error = np.mean(np.sum(np.square(data_predict-train_data), axis=0))

                if error < min_error:
                    min_error = error
                    water_cluster = cluster_

                print(cluster_)
                print('error as water endmember: {}'.format(error))

    print('water cluster = {}'.format(water_cluster))
    return water_cluster


def detectCluster (clusters_params, bands_keys, param, logic, band1, band2=None):

    param_list = []
    if band1:
        idx_band1 = bands_keys.index(band1)
    if band2:
        idx_band2 = bands_keys.index(band2)

    for clt in clusters_params:
        if param == 'diff':
            param_list.append(clt['mean'][idx_band1]-clt['mean'][idx_band2])
        elif param == 'value':
            param_list.append(clt['mean'][idx_band1])

    if logic == 'max':
        idx_detected = param_list.index(max(param_list))
    elif logic == 'min':
        idx_detected = param_list.index(min(param_list))

    return clusters_params[idx_detected]


def detectVegetationCluster(clusters_params, geo_bands):

    # if 'Red' in bands_keys:
    #     b2 = 'Red'
    #     if 'Nir2' in bands_keys:
    #         b1 = 'Nir2'
    #     elif 'Nir' in bands_keys:
    #         b1 = 'Nir'
    #     elif 'Mir' in bands_keys:
    #         b1 = 'Mir'
    #
    #     return detectCluster(clusters_params, bands_keys, 'diff', 'max', b1, b2)
    #
    # else:
    #     if 'Nir' in bands_keys:
    #         return detectCluster(clusters_params, bands_keys, 'value', 'max', 'Nir', None)
    #     if 'Nir2' in bands_keys:
    #         return detectCluster(clusters_params, bands_keys, 'value', 'max', 'Nir2', None)
    #     if 'Mir' in bands_keys:
    #         return detectCluster(clusters_params, bands_keys, 'value', 'max', 'Mir', None)

    bands_keys = list(sorted(geo_bands.keys()))
    return detectCluster(clusters_params, bands_keys, 'diff', 'max', 'Nir2', 'Red')


def findBestK(data, score_index='calinsk', min_k=3, max_k=6, train_size=1.0):

    # # split data for a smaller set (for performance purposes)
    # train_data, test_data = getTrainTestDataset(data, train_size, min_train_size=1000)

    if score_index == 'calinsk':
        print('Selection of best number of clusters using Calinski-Harabasz Index:')
    elif score_index == 'silhouete':
        print('Selection of best number of clusters using Silhouete Index:')

    computed_metrics = []

    for num_k in range(min_k, max_k+1):
        cluster_model = cluster.KMeans(n_clusters=num_k, init='k-means++')
        # cluster_model = GMM(n_components=num_k, covariance_type='full')
        # cluster_model = cluster.AgglomerativeClustering(n_clusters=num_k, linkage='ward')

        labels = cluster_model.fit_predict(data)

        if score_index == 'calinsk':
            computed_metrics.append(metrics.calinski_harabaz_score(data, labels))
            print('k={} :Calinski_harabaz index={}'.format(num_k, computed_metrics[num_k - min_k]))

        elif score_index == 'silhouete':
            computed_metrics.append(metrics.silhouette_score(data, labels))
            print('k={} :Silhouete index={}'.format(num_k, computed_metrics[num_k - min_k]))

    # the best solution is the one with higher index
    best_k = computed_metrics.index(max(computed_metrics)) + min_k

    return best_k


def applyKMeansCluster(data, k, train_size=1.0):

    # # split data not to train with all pixels (for performance purposes)
    # train_data, test_data = getTrainTestDataset(data, train_size, min_train_size=1000)

    cluster_model = cluster.KMeans(n_clusters=k, init='k-means++')
    cluster_model.fit(data)
    return cluster_model.predict(data)


def applyAglomerativeCluster(data, k, train_size=1.0):

    # # split data not to train with all pixels (for performance purposes)
    # train_data, test_data = getTrainTestDataset(data, train_size, min_train_size=5000, max_train_size=10000)

    cluster_model = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward')
    result = cluster_model.fit_predict(data)

    return result


def applyGaussianMixture(data, k, train_size=1.0):
    # # split data not to train with all pixels (for performance purposes)
    # train_data, test_data = getTrainTestDataset(data, train_size, min_train_size=5000, max_train_size=10000)
    model = GMM(n_components=k, covariance_type='full')
    model.fit(data)

    return model.predict(data)


def applyCluster(data, algorithm='kmeans', k=3, train_size=1.0):

    # before calling the clustering function, normalize the data using min_max_scaler
    # scaled_data = preprocessing.minmax_scale(data)
    scaled_data = data

    if algorithm == 'kmeans':
        return applyKMeansCluster(scaled_data, k, train_size)
    elif algorithm == 'aglomerative':
        return applyAglomerativeCluster(scaled_data, k, train_size)
    elif algorithm == 'gauss_mixture':
        return applyGaussianMixture(scaled_data, k, train_size)


def createMatriceCluster(base_array, indices_array, clusters_labels, water_cluster, veg_cluster):

    matrice_cluster = np.zeros_like(base_array)

    # apply water pixels to value 1
    matrice_cluster[indices_array[0][clusters_labels == water_cluster['clusterid']], indices_array[1][clusters_labels == water_cluster['clusterid']]] = 1

    # apply vegetation pixels to value 2
    matrice_cluster[indices_array[0][clusters_labels == veg_cluster['clusterid']], indices_array[1][clusters_labels == veg_cluster['clusterid']]] = 2

    # loop through the remaining labels and apply value >= 3
    new_label = 3
    for label_i in np.unique(clusters_labels):
        if (label_i != water_cluster['clusterid']) & (label_i != veg_cluster['clusterid']):
            matrice_cluster[indices_array[0][clusters_labels == label_i], indices_array[1][clusters_labels == label_i]] = new_label
            new_label += 1

    return matrice_cluster


############################################################################
# Graph functions
# -------------------------------------------------------------------------#
def plotClusteredData(data, cluster_names, file_name, graph_options):
    plt.style.use('seaborn-whitegrid')

    plot_colors = ['goldenrod', 'darkorange', 'tomato', 'brown', 'gray', 'salmon', 'black', 'orchid', 'firebrick']
    # plot_colors = list(colors.cnames.keys())

    fig, ax1 = plt.subplots()

    k = np.unique(data[:, 2])

    for i in k:
        cluster_i = data[data[:, 2] == i, 0:2]

        if int(i) in cluster_names.keys():
            label = cluster_names[int(i)]['name']
            colorname = cluster_names[int(i)]['color']
        else:
            label = 'Mixture'
            colorname = plot_colors[int(i)]

        ax1.set_xlabel(graph_options['x_label'])
        ax1.set_ylabel(graph_options['y_label'])
        ax1.set_title(graph_options['title'])

        ax1.plot(cluster_i[:, 0], cluster_i[:, 1], '.', label=label, c= colorname)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    plt.savefig(file_name + '.png')

    #plt.show()
    plt.close()

    return


def plotGraphs(bands, bands_combination, labels_array, file_name, invalid_mask=False, max_points=1000):

    # if combinations is not a list of lists, transform it in list of lists
    if type(bands_combination[0]) == str:
        bands_combination = [bands_combination]

    for bands_names in bands_combination:
        # O correto aqui e passar um dicionario com as opcoes, tipo, nome das legendas, etc.
        x_values = bands[bands_names[0]]
        y_values = bands[bands_names[1]]

        # create the graph filename
        graph_name = file_name + '_Graph_' + bands_names[0] + bands_names[1]

        # create the graph options dictionary
        graph_options = {'title': 'Scatterplot ' + bands_names[0] + ' x ' + bands_names[1],
                         'x_label': bands_names[0] + (' reflectance (' + dicS2BandNames[bands_names[0]] + ')' if bands_names[0] in dicS2BandNames else ''),
                         'y_label': bands_names[1] + (' reflectance (' + dicS2BandNames[bands_names[1]] + ')' if bands_names[1] in dicS2BandNames else '')}

        cluster_names = {1: {'name': 'Water', 'color': 'deepskyblue'},
                         2: {'name': 'Vegetation', 'color': 'forestgreen'}}

        # first, we will create the valid data array
        data = np.c_[x_values[~invalid_mask], y_values[~invalid_mask], labels_array[~invalid_mask]]

        plot_data, _ = getTrainTestDataset(data, train_size=1, min_train_size=0, max_train_size=max_points)

        plotClusteredData(plot_data, cluster_names, graph_name, graph_options)

    return


def plotPCA(bands, labels_array, file_name, invalid_mask=False, max_points=1000):

    data = bandsToColumns(bands, invalid_mask)
    labels = bandsToColumns({'labels': labels_array}, invalid_mask)

    pca = PCA(n_components=2).fit(data[:, 0:7])
    # pca = PCA().fit(data[:, 0:6])

    x_pca = np.c_[pca.transform(data[:, 0:7]), labels]

    x_pca, _ = getTrainTestDataset(x_pca, 1, 500, max_points)

    # create the graph options dictionary
    graph_options = {'title': 'PCA - 2 axis' ,
                     'x_label': 'Component 1',
                     'y_label': 'Component 2'}

    cluster_names = {1: {'name': 'Water', 'color': 'deepskyblue'},
                     2: {'name': 'Vegetation', 'color': 'forestgreen'}}

    plotClusteredData(x_pca, cluster_names, file_name + '_PCA', graph_options)
    # x_new = pca.inverse_transform(x_pca)

    return data


############################################################################
# calc INDEXES (MNDWI and NDVI)
# -------------------------------------------------------------------------#
def calcMndwi(bands, invalid_mask, min_mndwi):
    # calculate water index using green and mir bands
    mndwi = (bands['Green'] - bands['Mir2']) / (bands['Green'] + bands['Mir2'])

    mndwi_mask = invalid_mask | np.isinf(mndwi) | np.isnan(mndwi)

    mndwi[mndwi > 1] = 1
    mndwi[mndwi < -1] = -1

    mndwi = np.ma.array(mndwi, mask=mndwi_mask, fill_value=-9999)


    # calculate a water mask based on the water index
    water_mndwi_mask = mndwi.filled() > min_mndwi
    return mndwi, water_mndwi_mask


def calcNdwi(bands, invalid_mask):
    # calculate water index using green and mir bands
    ndwi = (bands['Green'] - bands['Nir2']) / (bands['Green'] + bands['Nir2'])

    ndwi_mask = invalid_mask | np.isinf(ndwi) | np.isnan(ndwi)

    ndwi[ndwi > 1] = 1
    ndwi[ndwi < -1] = -1

    ndwi = np.ma.array(ndwi, mask=ndwi_mask, fill_value=-9999)

    return ndwi


def calcNDVI(bands, invalid_mask):

    # calculate the NDVI index using red and nir bands
    ndvi = (bands['Nir2'] - bands['Red']) / (bands['Nir2'] + bands['Red'])

    ndvi_mask = invalid_mask | np.isinf(ndvi) | np.isnan(ndvi)

    ndvi[ndvi > 1] = 1
    ndvi[ndvi < -1] = -1

    ndvi = np.ma.array(ndvi, mask=ndvi_mask, fill_value=-9999)

    return ndvi


############################################################################
# Other utility functions
# -------------------------------------------------------------------------#
def getTrainTestDataset(data, train_size, min_train_size=10000, max_train_size=100000):
    dataset_size = data.shape[0]

    if (dataset_size * train_size) < min_train_size:
        train_size = min_train_size / dataset_size
        train_size = 1 if train_size > 1 else train_size

    elif (dataset_size * train_size) > max_train_size:
        train_size = max_train_size / dataset_size

    return train_test_split(data, train_size=train_size)


def createOutputDirectory(base_name, shp_name):
    # create the output folder begining with the shape name and after, the tile name
    shp_name, _ = os.path.splitext(shp_name)
    tile_name = os.path.split(base_name)[-1]
    base_name = os.path.join(os.path.split(base_name)[0], shp_name, tile_name)
    os.makedirs(base_name, exist_ok=True)
    base_name = os.path.join(base_name, tile_name)
    return base_name


def createComposite(bands, foldername, download='S2_THEIA'):

    # copy the RGB clipped bands to output directory

    redband = copy(bands['Red'].GetDescription(), foldername)
    greenband = copy(bands['Green'].GetDescription(), foldername)
    blueband = copy(bands['Blue'].GetDescription(), foldername)

    compositename = os.path.join(foldername, os.path.split(foldername)[-1]+'_composite.vrt')

    os.system('gdalbuildvrt -separate ' + compositename + ' ' +
              redband + ' ' + greenband + ' ' + blueband)

    return


def createProductName(bands_combination, clustering='aglomerative', classifier='naive_bayes', clip_mndwi=None):
    if clustering == 'aglomerative':
        product_name = 'AC_'
    elif clustering == 'gauss_mixture':
        product_name = 'GM_'
    elif clustering == 'kmeans':
        product_name = 'KM_'
    else:
        product_name = 'ERROR_'

    if classifier == 'naive_bayes':
        product_name += 'NB_'
    elif classifier == 'MLP':
        product_name += 'MLP_'
    elif classifier == 'Hull':
        product_name += 'Hull_'
    elif classifier == 'SVM':
        product_name += 'SVM'
    else:
        product_name += 'ERROR_'

    if clip_mndwi:
        product_name += 'CM_'
    for key in bands_combination:
        product_name += str(key)

    return product_name



