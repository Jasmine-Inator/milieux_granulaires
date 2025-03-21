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


def image_imports(path, templatepath,anim=False, rescale=True, scale=30, start=1): #use / in path
    img_list=[]
    pathlist=[]
    scale=int(scale)
    template=Image.open(templatepath)
    files={Path(filepath).stem: filepath for filepath in glob.glob(path)}
    for i, filename in tqdm(enumerate(glob.glob(path)), desc="Importing images"):
        j=i+start 
        name=f'25p ({j})'
        im=Image.open(files[name]).convert('RGB')
        pathlist.append(files[name])
        if not anim:
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
        coords_table.append(coordslist)
    return coords_table


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


def image_compare(images, modelpath, mass, Threshold=0.5):
    disttable=[]
    vectortable=[]
    speedtable=[]
    kinetictable=[]
    momentumtable=[]
    images=images.copy()
    centers_lists=palletcoords(pallet_check(images, modelpath), Threshold=Threshold)
    for i, centers in tqdm(enumerate(centers_lists), desc='comparing images'):
        try:
            cl1=centers
            cl2=centers_lists[i+1]
        except IndexError:
            return disttable, vectortable, speedtable, kinetictable
        distances, indexes=image_compare_dist(cl1,cl2)
        disttable.append(distances)
        vectors=image_compare_vect(cl1,cl2,indexes)
        vectortable.append(vectors)   
        vectors = image_compare_vect(cl1, cl2, indexes)
        speeds = compute_speed(vectors, delta_t=1)
        speedtable.append(speeds)
        kinetic_energies = compute_kinetic_energy(speeds, mass)
        kinetictable.append(kinetic_energies)
        momentumtable.append(mass*speeds)


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


def compute_speed(vectors, delta_t):
    speeds = np.linalg.norm(vectors, axis=1) / delta_t
    return speeds


def compute_kinetic_energy(speeds, mass):
    energies = 0.5 * mass * speeds**2
    return energies



path=("24_mm_25_particles/24_mm_25_partilces/*")
testpath=('Images/test/*')
templatepath=("Images/template.jpg")
scale=5
start=t.monotonic()
images=image_imports(path,templatepath, scale=scale)
modelpath=input('Put the relative path to exported model using / in the path')
tables=image_compare(images, modelpath)
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
