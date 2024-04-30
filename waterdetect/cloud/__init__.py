version = '0.0.1'

print('importing cloud package')
try:
    import planetary_computer
except: 
    print('Dependencies not found, please use `pip install waterdetect[cloud]` to install dependencies')
