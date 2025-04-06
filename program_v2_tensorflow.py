from tqdm import tqdm
#from celluloid import Camera
import time as t
import glob
import os
import numpy as np
import io
from six import BytesIO
from PIL import Image
from six.moves.urllib.request import urlopen
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
import scipy as sp
import itertools as it
import tensorflow as tf
from pathlib import Path
import moviepy
import math
import pandas as pd
import re

class imgdata:
    def __init__(self, positions, distances, vectors, timestamp, mass, framerate):
        self.positions=np.array(positions)
        self.distances=np.array(distances)
        self.vectors=np.array(vectors)
        self.speeds=np.array(self.compute_speed(framerate))
        self.kinetic_energies=np.array(self.compute_kinetic_energy(	mass))
        self.momentums=np.array(self.speeds*mass)
        self.timestamp=timestamp
        self.avg_distances=np.mean(self.distances)
        self.avg_speeds=np.mean(self.speeds)
        self.avg_kinetic_energies=np.mean(self.kinetic_energies)
        self.avg_momentums=np.mean(self.momentums)
        
    def compute_speed(self, framerate):
        speeds = np.linalg.norm(self.vectors, axis=1) / framerate
        return speeds
    
    def compute_kinetic_energy(self, mass):
        energies = 0.5 * mass * (self.speeds)**2
        return energies

class palletdata:
    def __init__(self, indexlist, positions, distances, vectors, mass, framerate, angthreshold=1):
        self.indexes=indexlist
        self.startindex=indexlist[0]
        self.vectors=np.array([vectors[i][index] for i, index in enumerate(indexlist)])
        self.positions=np.array([positions[i][index] for i, index in enumerate(indexlist)])
        self.distances=np.array([distances[i][index] for i, index in enumerate(indexlist)])
        self.speeds=self.compute_speed(framerate)
        self.kinetic_energies=self.compute_kinetic_energy(mass)
        self.momentums=self.speeds*mass
        self.avg_distances=np.mean(self.distances)
        self.total_distance=sum(self.distances)
        self.avg_speeds=np.mean(self.speeds)
        self.avg_kinetic_energies=np.mean(self.kinetic_energies)
        self.avg_momentums=np.mean(self.momentums)
        self.mean_free=self.mean_free_path(angthreshold)
        
    def compute_speed(self, framerate):
           speeds = np.linalg.norm(self.vectors, axis=1) / framerate
           return speeds
       
    def compute_kinetic_energy(self, mass):
           energies = 0.5 * mass * (self.speeds)**2
           return energies
    
    def mean_free_path(self, angthreshold):
        distances=self.distances[1:]
        vectors=self.vectors[1:]
        free_path=[]
        freedist=0
        for i, distance in enumerate(distances):
            try:
                v0=vectors[i]
                v1=vectors[i+1]
            except IndexError:
                meanfree=np.mean(np.array(free_path))
                return meanfree
            angle = abs(math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
            if angle <= angthreshold:
                freedist+=distance
            else:
                free_path.append(freedist)
                freedist=0

def image_imports(path, templatepath, n,docrop=True, rescale=True, scale=30, start=1): #use / in path
    img_list=[]
    pathlist=[]
    scale=int(scale)
    template=Image.open(templatepath)
    files=glob.glob(path)
    filedict={}
    for file in tqdm(files, desc= 'sorting files'):
        file=Path(file)
        stem=''
        filestem=file.stem
        for char in re.findall('[0-9]', filestem):
            stem+= char
        filedict.update({int(stem):file})
    filelabel=np.array(list(filedict.keys()))
    filelabel.sort()
    filelist=[filedict[label] for label in filelabel]
    for filename in tqdm(filelist, desc="Importing images"):
        im=Image.open(filename).convert('RGB')
        if  docrop:
            im=crop(im,template)
        if rescale:
            im=im.reduce(scale)
        im=im.rotate(-90)
        img_list.append(im)
    return img_list, pathlist


def crop(image,template, yoffset=25, xoffset=-15):
    width, height=template.size
    modwidth, modheight=image.size
    x1=modwidth/2+width/2+xoffset
    y1=modheight/2+height/2+yoffset
    x2=modwidth/2-width/2+xoffset
    y2=modheight/2-height/2+yoffset
    image=image.crop((x2,y2,x1,y1))
    return image

# keep working with tensorflow https://www.tensorflow.org/hub/tutorials/tf2_object_detection?hl=en
def pallet_check(images, modelpath):
    model=tf.saved_model.load(modelpath)
    images=images.copy()
    category_index = "Tensorflow/workspace/training_demo/annotations/label_map.pbtxt"
    convimages=[tf.image.convert_image_dtype(image, tf.uint8) for image in tqdm(images, desc='finishing to open images')]
    result_list=[]
    for im in tqdm(convimages, desc='checking for pallets'):
        im_height, im_width = im.size
        detections = model(im)
        result = {key:value.numpy() for key,value in detections.items()}
        result_list.append(result)
    return result_list, im_height, im_width


def palletcoords(result_list, im_height, im_width, Threshold = 0.5):
    result_list=result_list.copy()
    Threshold = Threshold
    coords_table=[]
    for results in tqdm(result_list, desc='processing results'):
        coordslist=[]
        bboxes = result_list['detection_boxes'][0].numpy()
        bscores = result_list['detection_scores'][0].numpy()
        for idx, boxes in tqdm(enumerate(bboxes)):
            if bscores[idx] >= Threshold:
                y_min = int(bboxes[idx][0] * im_height)
                x_min = int(bboxes[idx][1] * im_width)
                y_max = int(bboxes[idx][2] * im_height)
                x_max = int(bboxes[idx][3] * im_width)
                center=((x_min+x_max)/2, (y_min+y_max)/2)
                coordslist.append(center)
        coordslist=np.sort(np.array(coordslist),0)
        coords_table.append(coordslist)
    return coords_table


def scatter(coords_table, dirname='Frames', limits=(350,350)):
    coords_table=coords_table.copy()
    directory_name=dirname
    width, height=limits
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    plots=[]
    for i, coords_list in tqdm(enumerate(coords_table), desc='creating plots'):
        coords_list=flatten(coords_list)
        x = coords_list[::2]
        y = coords_list[1::2]
        fig = plt.figure()
        ax=fig.add_subplot()
        ax.set_aspect('equal', adjustable='box')
        plt.xlim(0,width)
        plt.ylim(0,height)
        #colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
        img_plot = plt.scatter(x,y,s=2, marker ='+')#, c=colors)
        plots.append(img_plot)
        figname='fig'+str(i)
        fig.savefig(f"{directory_name}/{figname}.png")
        plt.close()
    path=f'{directory_name}/*'
    return path, plots


def plotvector(origins, vectors):
    vectorcoords=flatten(vectors)
    origincoords=flatten(origins)
    vx=vectorcoords[::2]
    vy=vectorcoords[1::2]
    ox=origincoords[::2]
    oy=origincoords[1::2]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    vect_plot = plt.quiver(ox, oy, vx, vy)
    return vect_plot


def flatten(_list_):
    return np.array(list(it.chain.from_iterable(_list_)))


def cluster_find(coords_table, n):#need to make more accurate
    coords_table=coords_table.copy()
    cluster_list=[]
    for coords_list in tqdm(coords_table, desc='looking for clusters'):
        coords_list=np.array(coords_list)
        centroids, indexes= sk.cluster.kmeans_plusplus(coords_list, n)
        cluster_list.append(centroids)
    print('done looking for clusters')
    return cluster_list

def axis_coords_sort(array, axis):
    tags=[i for i in range(len(array))]
    adict={coord[axis]+10**(-len(str(len(array))))*tags[i]:coord[1-axis] for i, coord in enumerate(array)}
    keys=np.array(list(adict.keys()))
    keys.sort()
    newarray=np.array([np.array([int(key), adict[key]]) for key in keys])
    return newarray


def image_compare_dist(centers_lists, n, scale=11):
    centers_lists=centers_lists.copy()
    indextable=[np.array([i for i in range(n)])]
    disttable=[np.zeros(n)]
    for i, centers in tqdm(enumerate(centers_lists), desc='comparing distances'):
        try:
            cl1=centers
            cl2=centers_lists[i+1]
        except IndexError:
            disttable=np.array(disttable)
            print('disttable ok')
            indextable=np.array(indextable)
            print('indextable ok')
            return disttable, indextable
        center_list_1=np.array(cl1)
        center_list_2=np.array(cl2)
        distancestable=sp.spatial.distance.cdist(center_list_1, center_list_2)
        distances=[]
        indexes=[]
        for table in distancestable:
            table=table.tolist()
            distances.append(scale*min(table))
            indexes.append(table.index(min(table)))
        disttable.append(np.array(distances))
        indextable.append(np.array(indexes))


def image_compare_vect(center_list_1, center_list_2, neighbor_indexes):
    center_list_1=np.array(center_list_1)
    center_list_2=np.array(center_list_2)
    vectors=[]
    for i, point in tqdm(enumerate(center_list_1), desc='finding position vectors'):
        index=neighbor_indexes[i]
        neighbor=center_list_2[index]
        vectors.append(neighbor-point)
    vectors=np.array(vectors)
    return vectors


def image_compare(images, n, mass, framerate):
    n=n
    vectortable=[np.zeros((n,2))]
    images=images
    centers_lists=pallet_check(images)
    disttable, indexes=image_compare_dist(centers_lists, n)
    for i, centers in tqdm(enumerate(centers_lists), desc='comparing images'):
        try:
            cl1=centers
            cl2=centers_lists[i+1]
        except IndexError:
             images=[imgdata(centers_lists[i],disttable[i],vectortable[i], (i+1)/25, mass, framerate) for i, indexlist in enumerate(indexes)]
             avgimages=datasave(images)
             indexes=np.transpose(indexes)
             pallets=[palletdata(indexlist,centers_lists ,disttable, vectortable, mass, framerate) for indexlist in indexes]
             avgpallets=datasave(pallets)
             return images, pallets, avgimages, avgpallets, indexes 
        vectors=image_compare_vect(cl1,cl2,indexes[i])
        vectortable.append(vectors)

def images_to_video(image_folder,dirname,vidname='animation', fps=25):
    image_files=image_imports(image_folder, None, anim=True, start=0, rescale=False)
    #images.sort() 
    output_video_path=f'{dirname}'
    try:
        os.mkdir(output_video_path)
        print(f"Directory '{output_video_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{output_video_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{output_video_path}'.")
    if len(images) == 0:
        print("No image find in folder.")
        return
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{vidname}.mp4')
    os.rename(f'{vidname}.mp4',f'{output_video_path}/{vidname}.mp4' )
    print(f"Video successfully created  : {vidname}")
    return output_video_path

def datasave(datacollection):
    datacollection=datacollection[:]
    directory_name='saved_data'
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    try:
        t=[image.timestamp for image in datacollection]
        file='images.csv'
    except AttributeError:
        indexes=[pallet.startindex for pallet in datacollection]
        file='pallets.csv'
    distances=[obj.avg_distances for obj in datacollection]
    speeds=[obj.avg_speeds for obj in datacollection]
    kinetic_energies=[obj.avg_kinetic_energies for obj in datacollection]
    momentums=[obj.avg_momentums for obj in datacollection]
    try:
        fulldata=np.array([t[:],distances[:],speeds[:],kinetic_energies[:],momentums[:]])
        labels=['t','distance','speed','kinetic energy','momentum']
    except NameError:
        fulldata=np.array([indexes[:],distances[:],speeds[:],kinetic_energies[:],momentums[:]])
        labels=['index','distance between t and t-1','speed','kinetic energy','momentum']
    dataset = pd.DataFrame()
    for i, data in enumerate(fulldata):
        dataset[f'{labels[i]}']=fulldata[i]
    dataset.to_csv(f'saved_data/{file}')
    return dataset
        


path=("24_mm_25_particles/24_mm_25_partilces/*")
testpath=('Images/test/*')
templatepath=("Images/template.jpg")
scale=5
n=int(input('How many pallets are there?'))
start=t.monotonic()
images=image_imports(path,templatepath, scale=scale)
modelpath=input('Put the relative path to exported model using / in the path')
tables=image_compare(images, modelpath, n)
coordslist=palletcoords(pallet_check(images, modelpath))
#origins=cluster_find(coordslist, n)
image_folder = "Imageplots/*" 
output_video_path = "output_video"  
image_folder = "Imageplots/*" 
output_video_path = "output_video.mp4"  
images_to_video(image_folder, output_video_path, fps=25)
end=t.monotonic()
#disttable, vecttable=tables
#print(len(disttable), len(vecttable), end-start)
#scatterpath, plots= scatter(origins, scale=scale)
##plt.show()
##plots=[scatter(centers)]
##plt.show()
