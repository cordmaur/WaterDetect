from builtins import RuntimeError
import setuptools
import waterdetect
short_description = 'WaterDetect generates open water cover mask for L2A Sentinel 2 imagery without any a priori knowledge on the scene.'\
                    ' It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.'

long_description = short_description

setuptools.setup(
    name="waterdetect", # Replace with your own username
    version=waterdetect.__version__,
    author="Maurício Cordeiro",
    author_email="cordmaur@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cordmaur/WaterDetect",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['waterdetect=runWaterDetect:main'],
    },
    include_package_data=True,
    package_data={'waterdetect': ['../WaterDetect.ini', '../runWaterDetect.py']},
    install_requires=[
        'numpy>=1.17',
        'scikit_learn>=0.19',
        'matplotlib>=3.3',
        'PyPDF2>=1.26',
        'lxml>=4.5.0',
        'pillow>=7.0.0'
    ]
)