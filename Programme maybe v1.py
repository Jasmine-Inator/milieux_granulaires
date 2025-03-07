from PIL import Image, ImageOps
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools as it

def image_imports(path): #use / in path
    img_list=[]
    for filename in glob.glob(path):
        im=Image.open(filename).convert('RGB').rotate(-90)
        img_list.append(im)
    return img_list

def white_check(image):
    im=image
    width, height = im.size
    found_pixels = [i for i, pixel in enumerate(im.getdata()) if (pixel[0] + pixel[1] + pixel[2])/3 > (230)]
    found_pixels_coords = np.array([divmod(index, width) for index in found_pixels])
    return found_pixels_coords

def scatter(coords_list):
    x = np.array([coords[0] for coords in coords_list ])
    y = np.array([coords[1] for coords in coords_list ])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    img_plot = plt.scatter(x,y,s=2, marker ='+')
    return img_plot

def flatten(_list_):
    return np.array(list(it.chain.from_iterable(_list_)))

def cluster_find(coords_list, r=2):
    coords=[]
    cluster_coords=[]
    flat=flatten(coords)
    tree1 = sp.spatial.KDTree(coords_list, r)
    tree2 = tree1
    indexes=tree1.query_ball_tree(tree2, r)
    return indexes

def cluster_overlap(cluster_list, target=0.1):
    index=0
    overlap_index=[]
    cluster_list=cluster_list
    for cluster in cluster_list:
        weight=1/len(cluster)
        temp_cluster_list=cluster_list
        temp_cluster_list.remove(cluster)
        print (cluster, weight)
        for index in cluster:
            count = sum([index in cluster for (sub_temp_cluster_list) in temp_cluster_list])
        if weight*count > target:
            overlap_index.append(index)
        index+=1
    for clindex in overlap_index:
        cluster_list.remove(cluster_list[clindex])
    return cluster_list

def index_2_coords(index_list, coords_list):
    new_coords_list=[]
    for index in index_list:
        new_coords_list.append(coords_list[index])
    return new_coords_list





images=image_imports("Images/trimmed_picture.png")

#print([white_check(file)for file in images])
cl=[cluster_overlap(cluster_find((white_check(file)))) for file in images]
ind=[cluster_find((white_check(file))) for file in images][0]
print(cl, type(cl), len(ind), len(cl[0]))
plots = [scatter(index_2_coords(flatten(cl[0]), white_check(file))) for file in images]
plt.show()
#print(ind)
print (plots)