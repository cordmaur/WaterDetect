from builtins import RuntimeError
import setuptools

short_description = 'WaterDetect generates open water cover mask for L2A Sentinel 2 imagery without any a priori knowledge on the scene.'\
                    ' It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.'

try:
    # noinspection PyPackageRequirements
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except RuntimeError:
    long_description = short_description

setuptools.setup(
    name="waterdetect", # Replace with your own username
    version="1.0",
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
)