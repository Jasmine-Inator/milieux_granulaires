from PIL import Image
from tqdm import tqdm
#from celluloid import Camera
import time as t
import glob
import os
import moviepy
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import scipy as sp
import itertools as it
import sklearn as sk
from pathlib import Path
import math
import pandas as pd

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
            

def image_imports(path, templatepath, docrop=True, rescale=True, scale=30, start=1): #use / in path
    img_list=[]
    scale=int(scale)
    template=Image.open(templatepath)
    files=glob.glob(path)
    filedict={}
    for file in tqdm(files, desc= 'sorting files'):
        file=Path(file)
        stem=''
        filestem=file.stem
        for char in filestem:
            try:
                int(char)
                stem+=char
            except ValueError:
                pass
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

def white_check(images, save=True):
    images=images
    coords_list=[]
    for im in tqdm(images, desc='checking pixels'):
        width, height = im.size
        threshold=230
        found_pixels = [i for i, pixel in enumerate(im.getdata()) if (pixel[0]+pixel[1]+pixel[2])/3 > (threshold)]
        found_pixels_coords = [divmod(index, width) for index in found_pixels]
        coords_list.append(found_pixels_coords)
    if save:
        directory_name='Imageplots'
        scatter(coords_list, directory_name, limits=(width, height))
    return coords_list

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
        centroids=axis_coords_sort(centroids, 0)
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
    vectortable=[np.zeros((n,2))]
    images=images.copy()
    n=n
    coords=white_check(images, save=False)
    centers_lists=cluster_find(coords, n)
    disttable, indexes=image_compare_dist(centers_lists,n)
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
    files = [filepath for filepath in glob.glob(image_folder)]
    image_files=[]
    files={Path(filepath).stem: filepath for filepath in tqdm(glob.glob(image_folder), desc='importing frames')}
    for i, filename in tqdm(enumerate(glob.glob(path)), desc="sorting frames"):
        name=f'fig{i}'
        image_files.append(files[name])
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
        print("Aucune image trouvée dans le dossier.")
        return
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{vidname}.mp4')
    os.rename(f'{vidname}.mp4',f'{output_video_path}/{vidname}.mp4' )
    print(f"Vidéo créée avec succès : {vidname}")
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
        
    









path=('24_mm_50_particles_LRC/*')
testpath=('Images/test/*')
templatepath=("Images/template.jpg")
framerate=25
#width, height=Image.open('Images/pallet.jpg').size
scale=5
start=t.monotonic()
images=image_imports(path,templatepath, docrop=False, scale=scale)
n=50
mass=1
tables=image_compare(images, n, mass, framerate)
#impos=[image.positions for image in tables[0]]
#scatter(impos, dirname='images_newsort1')
#palpos=[pallet.positions for pallet in tables[1]]
#scatter(palpos, dirname='pallets_newsort1')
#coordslist=white_check(images, save=False)
#origins=cluster_find(coordslist, n)
#distances, indexes=image_compare_dist(origins)
#image_folder = "Imageplots/*" 
#output_video_path = "output_video"   
#images_to_video(image_folder, output_video_path, fps=25)
end=t.monotonic()
#disttable, vecttable=tables
#print(len(disttable), len(vecttable), end-start)
#scatterpath, plots= scatter(origins, scale=scale)
##plt.show()
##plots=[scatter(centers)]
##plt.show()
