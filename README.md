# WaterDetect
Water Detect Algorithm

The objective of this product is generate water masks automatically for Sentinel 2 and Landsat images.
The required libraries are:
GDAL>=3.0.2
snappy>=7.0.0
matplotlib>=3.1.2
PyPDF2>=1.26.0
scipy>=1.3.2
scikit-learn>=0.22
pysptools>=0.15.0



To use it, you should clone the project to your repository and run "python runWaterColor.py --help"

usage: runWaterColor.py [-h] -i INPUT -o OUT [-s SHP] [-p PRODUCT] [-g]
                        [-c CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The products input folder. Required.
  -o OUT, --out OUT     Output directory. Required.
  -s SHP, --shp SHP     SHP file. Optional.
  -p PRODUCT, --product PRODUCT
                        The product to be processed (S2_Theia, Landsat,
                        S2_L1C)
  -g, --off_graphs      Turns off the scatter plot graphs
  -c CONFIG, --config CONFIG
                        Configuration .ini file. If not specified
                        WaterDetect.ini is used as default

The input directory should contain the uncompressed folders for the images. The script will loop through all folders in the input directory and save the water masks, graphs and reports to the output folder. The output folder must be created beforehand.

If the config file is not specified, the script will search for WaterDetect.ini in the current folder.
