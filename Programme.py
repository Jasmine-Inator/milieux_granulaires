from PIL import Image
import glob
import numpy as np
def image_imports(path): #plese imput path as "yourpath/*"
    image_list= []
    for filename in glob.glob(path):
        im=Image.open(filename).convert('RGB')
        image_list.append(im)
    return image_list
def rgb_list(image):
    im=Image.open(image)
    pix_val = list(im.getdata())
    return pix_val
images=image_imports("C:/Users/Jas/Desktop/Projet PSC/Images/*")
print (images)
for file in images:
    print(rgb_list(file))