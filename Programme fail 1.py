from PIL import Image, ImageOps
import glob
import numpy as np

def rgb_lists(path): #use / in path
    images_rgbs=[]
    for filename in glob.glob(path):
        im=Image.open(filename).convert('RGB')
        im=ImageOps.flip(im)
        pix_val = list(im.getdata())
        images_rgbs.append(pix_val)
    return images_rgbs, path
#def pastille_check(image): #image is a list of rgb values
#    for rgb in image:

images=rgb_lists("C:/Users/Jas/Desktop/Projet PSC/Images/*")
print(images)
