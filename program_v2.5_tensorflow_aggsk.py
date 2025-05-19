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
import warnings
from sklearn.cluster import AgglomerativeClustering
warnings.filterwarnings("error")


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

def image_imports(path, templatepath, docrop=True, rescale=False, greyscale=False, scale=5, start=1): #use / in path
    img_list=[]
    scale=int(scale)
    template=Image.open(templatepath)
    files=glob.glob(testpath)
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
        im=Image.open(filename)
        if  docrop:
            im=crop(im,template)
        if rescale:
            im=im.reduce(scale)
        if greyscale:
            img=[]
            for pixel in im.getdata():
                grey=int((pixel[0]+pixel[1]+pixel[2])/3)
                img.append((grey,grey,grey))
            im.putdata(img)
        im=np.asarray(im)
        im=np.expand_dims(im, axis=0)
        img_list.append(im)
    template.close()
    return img_list


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

def pallet_check(images, modelpath, shape, show_boxes=True, save_dir='Detected'):
    model = tf.saved_model.load(modelpath)
    images = images.copy()
    im_height, im_width = images[0].shape[1:-1]
    convimages = [tf.image.convert_image_dtype(image, tf.uint8) for image in tqdm(images, desc='finishing to open images')]
    result_table = []

    if show_boxes:
        os.makedirs(save_dir, exist_ok=True)

    for i, im in tqdm(enumerate(convimages), desc='checking for pallets'):
        detections = model(im)
        result = {key: value.numpy() for key, value in detections.items()}
        result_table.append(result)

        if show_boxes:
            image_np_with_detections = im[0].numpy().copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                result['detection_boxes'][0],
                result['detection_classes'][0].astype(np.int32),
                result['detection_scores'][0],
                category_index={},  
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.35,
                agnostic_mode=True
            )
            out_path = os.path.join(save_dir, f"detection_{i:03}.jpg")
            Image.fromarray(image_np_with_detections).save(out_path)

    return result_table, im_height, im_width




def palletcoords(result_table, im_height, im_width, n):
    result_table=result_table.copy()
    coords_table=[]
    box_table=[]
    for i, results in tqdm(enumerate(result_table), desc='processing results'):
        boxlist=[]
        coordslist=[]
        bboxes = results['detection_boxes'][0]
        bscores = results['detection_scores'][0]
        scores=sorted(bscores)
        scores=set(scores[len(scores)-n:])
        Threshold=min(scores)
        #data={complex(score,i):box for i, (score, box) in enumerate(zip(bscores, bboxes))}
        bsizes=[(box[3]-box[1])*(box[2]-box[0]) for box in bboxes]
        sizes=sorted(bsizes)
        sizes=set(sizes[len(sizes)-n:])
        minsize=min(sizes)
        for idx, box in tqdm(enumerate(bboxes)):
                if bscores[idx] >= 0.35:
                    y_min = int(box[0] * im_height)
                    x_min = int(box[1] * im_width)
                    y_max = int(box[2] * im_height)
                    x_max = int(box[3] * im_width)
                    center=np.array([int((x_min+x_max)/2), int((y_min+y_max)/2)])
                    coordslist.append(center)
                    boxlist.append(box)
        box_table.append(boxlist)
        coords_table.append(coordslist)
    return coords_table



def aggmerge(coords_table, n, r):
    coords_table=coords_table.copy()
    centers_lists=[]
    for coordlist in tqdm(coords_table, desc='Merging excess points'):
        if len(coordlist)>n:
            tags=[i for i in range(len(coordlist))]
            agg = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=r)
            labels = agg.fit_predict(coordlist)
            data={labels[i]+10**(-len(str(len(coordlist))))*tags[i]:coords for i, coords in enumerate(coordlist)}
            groups=[]
            for i in range(n):
                group=[]
                for label in data:
                    if int(label)==i:
                        group.append(data[label])
                groups.append(np.array(group))
            centers=[]
            for group in groups:
                if len(group) != 0:
                    fgroup=flatten(group)
                    groupx=fgroup[::2]
                    groupy=fgroup[1::2]
                    center=np.array([np.mean(groupx), np.mean(groupy)])
                    centers.append(center)
            centers_lists.append(centers)
        centers_lists.append(coordlist)
    return centers_lists

def scatter(coords_table, dirname='Frames', limits=(350,350), track=False):
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
        img_plot = plt.scatter(x,y,s=2, marker ='+')
        if track:
            plt.plot(x,y, linewidth=1.5)
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



def axis_coords_sort(array, axis):
    tags=[i for i in range(len(array))]
    try:
        array=array.tolist()
    except AttributeError:
        pass
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


def image_compare(images, n, modelpath, shape, mass=1, framerate=25, radius=60):
    diameter=2*radius
    n=n
    vectortable=np.array([np.zeros((n,2))])
    images=images
    result_list, im_height, im_width=pallet_check(images, modelpath,shape)
    centers_lists=aggmerge(palletcoords(result_list, im_height, im_width, n), n, radius)
    disttable, indexes=image_compare_dist(centers_lists,n)
    for i, centers in tqdm(enumerate(centers_lists), desc='comparing images'):
        try:
            cl1=centers
            cl2=centers_lists[i+1]
        except IndexError:
             scatter(centers_lists,limits=(im_width,im_height))
             images=[imgdata(centers_lists[i],disttable[i],vectortable[i], (i+1)/25, mass, framerate) for i, indexlist in enumerate(indexes)]
             avgimages=datasave(images)
             indexes=np.transpose(indexes)
             pallets=[palletdata(indexlist,centers_lists ,disttable, vectortable, mass, framerate) for indexlist in indexes] 
             positions=[pallet.positions for pallet in pallets] 
             scatter(positions, limits=(im_width,im_height), dirname='Pallets', track=True)
             avgpallets=datasave(pallets)
             print(positions)
             #images_to_video('Frames', 'animation')
             return images, pallets, avgimages, avgpallets, indexes 
        vectors=image_compare_vect(cl1,cl2,indexes[i])
        vectortable=vectortable.tolist()
        vectortable.append(vectors)
        vectortable=np.array(vectortable)

def images_to_video(image_folder,dirname,vidname='animation', fps=25):
    image_files=image_imports(image_folder, None, start=0, rescale=False)
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
        mean_free=[pallet.mean_free for pallet in datacollection]
    distances=[obj.avg_distances for obj in datacollection]
    speeds=[obj.avg_speeds for obj in datacollection]
    kinetic_energies=[obj.avg_kinetic_energies for obj in datacollection]
    momentums=[obj.avg_momentums for obj in datacollection]
    try:
        fulldata=np.array([t[:],distances[:],speeds[:],kinetic_energies[:],momentums[:]])
        labels=['t','distance','speed','kinetic energy','momentum']
    except NameError:
        fulldata=np.array([indexes[:],distances[:],speeds[:],kinetic_energies[:],momentums[:], mean_free[:]])
        labels=['index','distance between t and t-1','speed','kinetic energy','momentum', 'mean free path']
    dataset = pd.DataFrame()
    for i, data in enumerate(fulldata):
        dataset[f'{labels[i]}']=fulldata[i]
    dataset.to_csv(f'saved_data/{file}')
    return dataset
        
def setup():
    path=input('Imput the path to the folder with the images.')
    path=path+'/*'
    if os.path.isdir(path)!=True:
        print('Invalid path')
        exit()
    shape=input('Which shape do you want to track (use _ instead of spaces)')
    modelpath=f'Tensorflow/workspace/training_demo_{shape}/exported_models/my_model/saved_model'
    if os.path.isdir(modelpath)!=True:
        print('Shape not handeled')
        exit()
    n=abs(int(input('How many pallets are there?')))
    radius=abs(int(input('What is the radius of the pallets (in pixels)?')))
    return path, modelpath, n, radius

#path, modelpath, n, radius = setup()
path=("24_mm_25_particles/24_mm_25_particles/*")
testpath=("Images/test/*")
templatepath=("Images/template.jpg")
#♣n=int(input('how many pallets are there?'))
n=25
radius=60
start=t.monotonic()
images=image_imports(testpath, templatepath, rescale=False, docrop=True)
#shape=input('Which shape do you want to track (use _ instead of spaces)')
shape='circle'
modelpath=f'Tensorflow/workspace/training_demo_{shape}/exported-models/my_model/saved_model'
tables=image_compare(images,n, modelpath,shape, radius=radius)
#result_list, im_height, im_width=pallet_check(images, modelpath,shape)
#coordslist=(palletcoords(result_list, im_height, im_width, n))
#scatter(coordslist, limits=(im_height,im_width))
#dia=60
#merged=merge2(coordslist, dia, n)
#print(merged[1], [len(image) for image in merged] )
#scatter(merged, limits=(im_height,im_width), dirname='Merged')
end=t.monotonic()
#disttable, vecttable=tables
#print(len(disttable), len(vecttable), end-start)
#scatterpath, plots= scatter(origins, scale=scale)
##plt.show()
##plots=[scatter(centers)]
##plt.show()
