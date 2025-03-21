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


def image_imports(path, templatepath,anim=False, rescale=True, scale=30): #use / in path
    img_list=[]
    scale=int(scale)
    template=Image.open(templatepath)
    files={Path(filepath).stem: filepath for filepath in glob.glob(path)}
    for i, filename in tqdm(enumerate(glob.glob(path)), desc="Importing images"):
        j=i+1 
        name=f'25p ({j})'
        im=Image.open(files[name]).convert('RGB')
        if not anim:
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

def white_check(images, scale=30):
    images=images
    coords_list=[]
    directory_name='Imageplots'
    for im in tqdm(images, desc='checking pixels'):
        width, height = im.size
        threshold=230
        found_pixels = [i for i, pixel in enumerate(im.getdata()) if (pixel[0]+pixel[1]+pixel[2])/3 > (threshold)]
        found_pixels_coords = [divmod(index, width) for index in found_pixels]
        coords_list.append(found_pixels_coords)
    scatter(coords_list, directory_name, scale=scale)
    return coords_list

def scatter(coords_table, dirname='Frames', scale=30):
    coords_table=coords_table.copy()
    directory_name=dirname
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
        plt.xlim(0,1747/scale)
        plt.ylim(0,1747/scale)
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

def image_compare_dist(center_list_1, center_list_2, neighbor_indexes=True, scale=11):
    indexes=[]
    distances=[]
    center_list_1=np.array(center_list_1)
    center_list_2=np.array(center_list_2)
    distancestable=sp.spatial.distance.cdist(center_list_1, center_list_2)
    for table in tqdm(distancestable, desc='finding shortest distances'):
        table=table.tolist()
        distances.append(scale*min(table))
        indexes.append(table.index(min(table)))
    if neighbor_indexes:
        return distances, indexes
    return distances


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


def image_compare(images, n):
    disttable=[]
    vectortable=[]
    speeds=[]
    kinetic_energies=[]
    images=images
    n=n
    coords=white_check(images)
    centers_lists=cluster_find(coords, n)
    for i, centers in tqdm(enumerate(centers_lists), desc='comparing images'):
        try:
            cl1=centers
            cl2=centers_lists[i+1]
        except IndexError:
            return disttable, vectortable, speeds, kinetic_energies
        distances, indexes=image_compare_dist(cl1,cl2)
        disttable.append(distances)
        vectors=image_compare_vect(cl1,cl2,indexes)
        vectortable.append(vectors)
        vectors = image_compare_vect(cl1, cl2, indexes)
        speeds.append(compute_speed(vectors, delta_t=1))  
        mass = 1.0
        kinetic_energies.append(compute_kinetic_energy(speeds, mass))


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
    os.remove(f'{output_video_path}/{vidname}.mp4')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{vidname}.mp4')
    os.rename(f'{vidname}.mp4',f'{output_video_path}/{vidname}.mp4' )
    print(f"Vidéo créée avec succès : {vidname}")
    return output_video_path


def compute_speed(vectors, delta_t):
    speeds = np.linalg.norm(vectors, axis=1) / delta_t
    return speeds


def compute_kinetic_energy(speeds, mass):
    energies = 0.5 * mass * speeds**2
    return energies








path=("24_mm_25_particles/24_mm_25_partilces/*")
testpath=('Images/test/*')
templatepath=("Images/template.jpg")
width, height=Image.open('Images/pallet.jpg').size
scale=5
start=t.monotonic()
images=image_imports(path,templatepath, scale=scale)
n=25
#tables=image_compare(images, n)
coordslist=white_check(images, scale=scale)
#origins=cluster_find(coordslist, n)
image_folder = "Imageplots/*" 
output_video_path = "output_video"  
origins=cluster_find(coordslist, n)
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
