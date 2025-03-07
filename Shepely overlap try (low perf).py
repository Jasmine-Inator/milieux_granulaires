from PIL import Image, ImageOps
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools as it
import shapely as sh

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
    print(coords_list)
    coords_list=flatten(coords_list)
    x = coords_list[::2]
    y = coords_list[1::2]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    img_plot = plt.scatter(x,y,s=2, marker ='+')
    return img_plot

def flatten(_list_):
    return np.array(list(it.chain.from_iterable(_list_)))

def cluster_find(coords_list, r=2):
    print('looking for clusters')
    tree1 = sp.spatial.KDTree(coords_list, r)
    tree2 = tree1
    indexes=tree1.query_ball_tree(tree2, r)
    print('done looking for clusters')
    return indexes

def cluster_exclusion(cluster_list, margin=0):
    print('excluding clusters that are too small')
    criteria=len(max(cluster_list, key=len))-(len(max(cluster_list, key=len))*margin)
    new_cluster_list=[cluster for cluster in cluster_list if len(cluster)>=criteria]
    print((len(new_cluster_list)/len(cluster_list))*100, 'clusters kept')
    print('done excluding clusters')
    return new_cluster_list

def cluster_overlap(cluster_list, coords_list): #Need to speed up
    print('looking for overlapping clusters')
    indexes=[]
    for i, cluster in enumerate(np.array(cluster_list)):
        cluster=sh.multipoints(index_2_coords(cluster, coords_list))
        for temp_cluster in cluster_list:
            temp_cluster=sh.multipoints(index_2_coords(temp_cluster, coords_list))
            if cluster == temp_cluster:
                continue
            sh.prepare(cluster)
            sh.prepare(temp_cluster)
            if sh.overlaps(cluster, temp_cluster) and i not in indexes: #and not sh.touches(cluster, temp_cluster)
                indexes.append(i)
                break
        print((i/len(cluster_list))*100,"% done")
    print('done looking for overlapping clusters')
    return indexes

def overlap_remove(index_list, cluster_list):
    print('removing overlapping clusters')
    for index in index_list:
        cluster_list.remove(cluster_list[index])
    print('done removing overlapping clusters')
    return cluster_list

def index_2_coords(index_list, coords_list):
    new_coords_list=[]
    for index in index_list:
        new_coords_list.append((coords_list[index].tolist()))
    return np.array(new_coords_list)

def cluster_center(cluster_list, coords_list):
    centers=[]
    print('Looking for centers')
    for cluster in cluster_list:
        cluster=index_2_coords(cluster, coords_list)
        cluster=sh.MultiPoint(cluster)
        centers.append(flatten((cluster.centroid).coords))
    print('done looking for centers')
    return np.array(centers)


images=image_imports("Images/trimmed_picture.png")
r=40
coords=[white_check(file)for file in images][0]
clusters=cluster_exclusion(cluster_find(coords))
#cl=[cluster_center((cluster_find((white_check(file)))), coords) for file in images]
#ind=[cluster_find((white_check(file))) for file in images][0]
#plots = [scatter(cl[0]) for file in images]
#plt.show()
#plot=scatter(([white_check(file)for file in images][0]))
#plt.show()
#print(ind)
#print (plots)
index=cluster_overlap(clusters,coords)
print(index)
coord=index_2_coords((flatten(overlap_remove(index, clusters))), coords)
print(coord)
plot=scatter(coord)
plt.show()