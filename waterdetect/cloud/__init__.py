from .imgfinder import RSImageFinder
from .rsimage import RSImage
from .s2 import S2Image
from .l8 import L8Image


version = "0.0.1"

print("importing cloud package")
try:
    import planetary_computer
except:
    print(
        "Dependencies not found, please use `pip install waterdetect[cloud]` to install dependencies"
    )
