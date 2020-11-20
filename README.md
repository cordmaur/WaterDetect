# WaterDetect

[![DOI](https://zenodo.org/badge/224832878.svg)](https://zenodo.org/badge/latestdoi/224832878)

## Synopsis

WaterDetect is an end-to-end algorithm to generate open water cover mask, specially conceived for L2A Sentinel 2 imagery from [MAJA](https://logiciels.cnes.fr/en/content/maja)<sup>1</sup>  processor, without any a priori knowledge on the scene. It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.<br>

The water masks produced by WaterDetect were primarily designed for water quality product computation (Obs2Co processing chain) and are also used for multi-temporal water maps (Surfwater processing chain). Both chains are supported by the "SWOT-Downstream" and TOSCA programs by CNES. Products are provided by the THEIA / Hydroweb-NG platform. 

The WaterDetect algorithm uses a multidimensional agglomerative clustering technique on a subsample of the scene's pixels, to group them in classes, and a naive bayes classifier to generalize the results for the whole scene, as summarized in the following picture:

![Screenshot](GraphicalAbstract.JPG)

All the details and tests has been described in the article <b>Automatic Water Detection from Multidimensional Hierarchical Clustering for Sentinel-2 Images and a Comparison with Level 2A Processors</b>, under revision by the journal Remote Sensing of Environment.

## How to cite
<b> (Accepted by Remote Sensing of Environment, pending publication):</b><br>
Cordeiro, M.C.R, Martinez, J.-M., Pena Luque, S., 2020. Automatic Water Detection from Multidimensional Hierarchical Clustering for Sentinel-2 Images and a Comparison with Level 2A Processors. Remote Sensing of Environment XX, XX. 

## Supported Formats
The algorithm has been developed taking into account atmospherically corrected images from MAJA, as described in the paper. However other image formats are also supported.
To the present, the following image formats are supported:
* Sentinel 2 - L2A from MAJA: the products can be downloaded from (https://www.theia-land.fr/en/product/sentinel-2-surface-reflectance/)
* Sentinel 2 - L2A from Sen2Cor: The L2A processed by Sen2Cor are available at Copernicus SciHub (https://scihub.copernicus.eu/)
* Sentinel 2 - L1C: L1C Sentinel 2 images can be downloaded from Copernicus SciHub (https://scihub.copernicus.eu/)
* Landsat 8 - To be validated

## Dependencies
The required libraries are:
```
GDAL>=3.0.2
matplotlib>=3.1.2
PyPDF2>=1.26.0
scipy>=1.3.2
scikit-learn>=0.22
skikit-image>=0.16.2
numpy>=1.17
PIL>=8.0
lxml>=4.5
```
### Note 1:
GDAL is required to open the satellite images. It's still possible to use without GDAL, from a python console or jupyter notebook, loading the rasters manually and passing all the necessary bands to the DWImageClustering class. Check the topic "Usage from Console" for more information.

### Note 2:
Scikit-Image is only necessary to run Otsu threshold method. 

The test_dependencies.py can be used to check if all libraries are loading correctly. Simply run:


## Instalation
The easiest way to install waterdetect package is with `pip` command:<br>
`pip install waterdetect`

Alternatively, you can clone the repository and install from its root throught the following commands:
```
git clone https://github.com/cordmaur/WaterDetect.git
cd WaterDetect
pip install .
```

Once installed, a `waterdetect` entry point is created in the path of the environment.
One can check the installation and options by running `waterdetect --help`. If GDAL is not found, a message will raise indicating that waterdetect will only run from a console.
```
usage: waterdetect [-h] [-GC] [-i INPUT] [-o OUT] [-s SHP] [-p PRODUCT]
                   [-c CONFIG]

The waterdetect is a high speed water detection algorithm for satellite
images. It will loop through all images available in the input folder and
write results for every combination specified in the .ini file to the output
folder. It can also run for single images from Python console or Jupyter
notebook. Refer to the onlinedocumentation

optional arguments:
  -h, --help            show this help message and exit
  -GC, --GetConfig      Copy the WaterDetect.ini from the package into the
                        specifieddirectory and skips the processing. Once
                        copied you can edit the .ini file and launch the
                        waterdetect without -c option.
  -i INPUT, --input INPUT
                        The products input folder. Required.
  -o OUT, --out OUT     Output directory. Required.
  -s SHP, --shp SHP     SHP file. Optional.
  -p PRODUCT, --product PRODUCT
                        The product to be processed (S2_THEIA, L8_USGS, S2_L1C
                        or S2_S2COR)
  -c CONFIG, --config CONFIG
                        Configuration .ini file. If not specified
                        WaterDetect.ini from current dir and used as default

To copy the package's default .ini file into the current directory, type:
`waterdetect -GC .` without other arguments and it will copy WaterDetect.ini
into the current directory.
```

### Config File
The waterdetect needs a config file that specifies the bands used in the clustering process as well as other parameters.
To obtain the default version of this file,  one can use `waterdetec -GC` and the file WaterDetect.ini will be copied into the current working folder.

## Usage as Script
The basic usage for the waterdetect is:<br>
`waterdetect -i c:/input_folder -i -c:/output_folder -p S2_THEIA`

The input directory should contain the uncompressed folders for the images. The script will loop through all folders in the input directory and save the water masks, graphs and reports to the output folder. The output folder must be created beforehand.

If the config file is not specified, the script will search for WaterDetect.ini in the current folder.

## Usage from Console
Once properly installed, the WaterDetect can be run from a JupyterNotebook:



## Contributors
> Author: Maurício Cordeiro (ANA/GET)<br>
> Supervisor: Jean-Michel Martinez (IRD/GET)<br>
> Validation dataset: Santiago Pena Luque (CNES) 

### Institutions
* ANA - Agência Nacional de Águas (https://www.gov.br/ana/en/)
* GET - Géosciences Environnement Toulouse (https://www.get.omp.eu/)
* IRD - Institut de Recherche pour le Développement (https://en.ird.fr/)
* CNES - Centre National d'Études Spatiales (https://cnes.fr/fr)

## License
This code is licensed under the [GNU General Public License v3.0](https://github.com/cordmaur/WaterDetect/blob/master/LICENSE) license. Please, refer to GNU's webpage  (https://www.gnu.org/licenses/gpl-3.0.en.html) for details.

## Reference
(1) Hagolle, O.; Huc, M.; Pascual, D. V.; Dedieu, G. A Multi-Temporal Method for Cloud Detection, Applied to FORMOSAT-2, VENµS, LANDSAT and SENTINEL-2 Images. Remote Sensing of Environment 2010, 114 (8), 1747–1755. https://doi.org/10.1016/j.rse.2010.03.002.
