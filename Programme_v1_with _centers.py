from PIL import Image, ImageOps
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools as it
import shapely as sh

class indexlist:
    def __init__(self, index_list):
        self.indexes = index_list
        try:
            self.len=len(index_list)
        except AttributeError:
            pass


    def ascoords(self, coords_list):
        index_list=self.indexes
        return index_2_coords(index_list, coords_list)

class cluster(indexlist):
    def __init__(self, indexes, coords_system):
        super().__init__(indexlist)
        self.ref=coords_system
        self.indexeslist=indexlist(indexes)
        self.indexes=indexes
        self.coords=self.ascoords(self.ref)
        self.points=[sh.geometry.Point(point) for point in self.coords]
        self.geom=sh.MultiPoint(self.points)
        try:
            self.polygon=sh.geometry.Polygon([(point.x, point.y) for point in self.points])
        except ValueError:
            self.polygon=sh.MultiPoint(self.points)
                try:
            self.sorted=np.sort(np.array(indexes))
        except AttributeError:
            pass
        self.center=flatten(((self.geom).centroid).coords)


def image_imports(path): #use / in path
    img_list=[]
    for filename in glob.glob(path):
        im=Image.open(filename).convert('RGB').rotate(-90)
        img_list.append(im)
    return img_list

def white_check(image):
    im=image
    width, height = im.size
    found_pixels = [i for i, pixel in enumerate(im.getdata()) if (pixel[0] + pixel[1] + pixel[2])/3 > (210)]
    found_pixels_coords = np.array([divmod(index, width) for index in found_pixels])
    return found_pixels_coords

def scatter(coords_list):
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
    tree1 = sp.spatial.KDTree(coords_list)
    tree2 = tree1
    indexes=tree1.query_ball_tree(tree2, r)
    indexes=[cluster(index_collection, coords_list) for index_collection in indexes]
    print('done looking for clusters')
    return indexes

##def cluster_exclusion(cluster_list, margin=0.1, crtype='a', invert=False): # change to only keep one of each cluster size
##    print('excluding clusters that are too small')
##    if crtype=='a':
##        criteria=np.average([sh.area(cluster.polygon) for cluster in cluster_list if sh.area(cluster.polygon) != 0])
##    elif crtype == 'min':
##        criteria=min([sh.area(cluster.polygon) for cluster in cluster_list if sh.area(cluster.polygon) != 0])
##    elif crtype=='max':
##        criteria=max([sh.area(cluster.polygon) for cluster in cluster_list if sh.area(cluster.polygon) != 0])
##    indexes=[]
##    for i, cluster in enumerate(cluster_list):
##        if sh.area(cluster.polygon) < criteria*(1-margin) or sh.area(cluster.polygon) > criteria*(1+margin):
##            indexes.append(int(i))
##    if not invert:
##        indexes=[index for index, cluster in enumerate(cluster_list) if index not in indexes]
##    new_cluster_list=cluster_remove(indexes, cluster_list)
##    print(len(new_cluster_list)/len(cluster_list),len(new_cluster_list), len(cluster_list) )
##    print('done excluding clusters')
##    return new_cluster_list

def group(center_list, r=2):
    center_list=flatten(center_list)
    x = coords_list[::2]
    y = coords_list[1::2]
    for i, y in enumerate(y):
        merged_xs=[x for x in x if ]
    new_x.append(np.average(np.array[merged_xs]))

def cluster_fusion(cluster_list, margin=0.1, r=2):
    indexes=[]
    fused_indexes=[]
    center_list=clusters_center(cluster_list)
    tree1=sp.spatial.KDTree(center_list)
    for center in center_list:
        indexes.append(tree1.query_ball_point(center, r-(r*margin)))
    indexes=[cluster(index_collection, center_list) for index_collection in indexes]




def cluster_overlap(center_list, margin=0.1, r=2): #need to make more accurate
    print('looking for overlaps')
    indexes=[]
    tree1=sp.spatial.KDTree(center_list)
    for center in center_list:
        indexes.append(tree1.query_ball_point(center, r-(r*margin)))
    indexes=[cluster(index_list, center_list) for index_list in indexes]
    print('done looking for overlaps')
    return indexes

def cluster_remove(index_list, cluster_list):
    print('removing clusters')
    try:
        index_list=[cluster.indexes for cluster in index_list]
    except AttributeError:
        pass
    try:
        index_list=(list(set(flatten(index_list))))
    except TypeError:
        pass
    index_list=np.array(index_list)
    new_cluster_list=np.delete(cluster_list, index_list, axis=0)
    print('done removing  clusters')
    return new_cluster_list

def index_2_coords(index_list, coords_list):
    new_coords_list=[]
    try:
        index_list.indexes
    except AttributeError:
        pass
    for index in index_list:
        new_coords_list.append(np.array(coords_list[index]))
    return np.array(new_coords_list)

def clusters_center(cluster_list):
    centers=[]
    print('Looking for centers')
    for cluster in cluster_list:
        centers.append(cluster.center)
    print('done looking for centers')
    return np.array(centers)

##def true_center(cluster_list, margin=0.1, r=2):
##    center_list=clusters_center(cluster_list)
##    true_centers=clusters_center(cluster_exclusion(cluster_overlap(center_list, margin, 0.5*r), 0.5, crtype='a', invert=True))
##    return true_centers




images=image_imports("Images/trimmed_picture_copy.png")
r=10
coords_list=[white_check(file)for file in images][0]
print(coords_list)
clusters=cluster_exclusion((cluster_find(coords_list, r)))
#cl=[cluster_center((cluster_find((white_check(file)))), coords) for file in images]
#ind=[cluster_find((white_check(file))) for file in images][0]
#plots = [scatter(cl[0]) for file in images]
#plt.show()
#plot=scatter(([white_check(file)for file in images][0]))
#plt.show()
#print(ind)
#print (plots)
centers=clusters_center(clusters)
index=cluster_overlap(centers, r)
print(len(index))
coord=index_2_coords((flatten(cluster_remove(index, centers))), coords_list)
print(coord, len(coords_list), len(centers), len(clusters))
plot=[scatter(centers), scatter(coord), scatter(coords_list)]
plt.show()
