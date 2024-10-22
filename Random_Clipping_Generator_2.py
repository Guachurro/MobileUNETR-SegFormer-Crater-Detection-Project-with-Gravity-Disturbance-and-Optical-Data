# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:57:33 2024

@author: Guach
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:05:00 2024

@author: Guach
"""
import geopandas as gpd

import pandas as pd

import rasterio

from shapely import Point, Polygon

from matplotlib import pyplot as plt

from rasterio import mask

from rasterio.windows import Window

import numpy

#import torchvision.transforms as T

#import pyproj

#from pyproj import Geod

import random

from PIL import Image

import os

from numpy import nan
#Full="F:/Academics/OSU/Thesis Documents/Lunar Gravity Maps/gggrx_1200a_dist_l1200.tif"

maptype=1200
CentLon='13E'
CentLat='0N'
Dataset='1'

#A clip of the Label and the full Image map 
GravLabel=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {Dataset}/{CentLon}_{CentLat}_30.0by30.0Area_{maptype}_Label.tif"
GravClip=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {Dataset}/{CentLon}_{CentLat}_30.0by30.0Area_{maptype}.tif"

OutputDest=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {Dataset}/{maptype}"



#Size in terms of Degrees
min_size=4 #This is about 18km in the equator. 
max_size=60 # This is 4 times bigger than the biggest crater we've allowed (70km)
num_clippings=600 #We will want as many as we can manage. 1000 Where the minimum size can be at least 2 will provide good local and global examples. 

#To save clippings
def SaveImage(Dataset,image,out_image,out_transform,Output,number,label=False, augmentation=False,rotation=0,maptype=0):
    Image1=rasterio.open(image)
    meta=Image1.profile
    meta.update({"driver":Image1.driver,
                 "height":out_image.shape[1],
                 "width":out_image.shape[2],
                 "count":out_image.shape[0],
                 "transform":out_transform
                 })
    #This If saves original and augmented Images
    if label==False:          
        filename=os.path.join(Output,'Image/',f"Set_{Dataset}_Im_{number+1}_D{maptype}.tif")
        with rasterio.open(filename, "w",**meta) as dest:
            dest.write(out_image)
            
        if augmentation==True:
            #HORIZONTAL FLIP
            out_image1=numpy.squeeze(out_image)
            out_image1=numpy.fliplr(out_image1)
            out_image1=numpy.expand_dims(out_image1,axis=0)
            meta.update({
                "driver":Image1.driver,
                "height":out_image1.shape[1],
                "width":out_image1.shape[2],
                "count":out_image1.shape[0],
                "transform":out_transform
                })
            filename1=os.path.join(Output,'Image/',f"Set_{Dataset}_Im_{number+1}_HorizontalFlip_D{maptype}.tif")
            with rasterio.open(filename1, "w",**meta) as dest:
                dest.write(out_image1)
            
            #VERTICAL FLIP
            out_image2=numpy.squeeze(out_image)
            out_image2=numpy.flipud(out_image2)
            out_image2=numpy.expand_dims(out_image2,axis=0)
            meta.update({
                "driver":Image1.driver,
                "height":out_image2.shape[1],
                "width":out_image2.shape[2],
                "count":out_image2.shape[0],
                "transform":out_transform
                })
            filename2=os.path.join(Output,'Image/',f"Set_{Dataset}_Im_{number+1}_VerticalFlip_D{maptype}.tif")
            with rasterio.open(filename2, "w",**meta) as dest:
                dest.write(out_image2)
                
            
            #Rotation
            out_image4=numpy.squeeze(out_image)
            pilimage=Image.fromarray(out_image4)
            rotated_image=pilimage.rotate(rotation)
            out_image4=numpy.array(rotated_image)
            out_image4=numpy.expand_dims(out_image4,axis=0)                
            meta.update({
                "driver":Image1.driver,
                "height":out_image4.shape[1],
                "width":out_image4.shape[2],
                "count":out_image4.shape[0],
                "transform":out_transform
                })
            filename4=os.path.join(Output,'Image/',f"Set_{Dataset}_Im_{number+1}_Rotated_{rotation}dg_D{maptype}.tif")
            with rasterio.open(filename4, "w",**meta) as dest:
                dest.write(out_image4)

            
    #This will save original and augmented labels
    else:
        filename=os.path.join(Output,'Label/',f"Set_{Dataset}_Im_{number+1}_D{maptype}_Label.tif")
        with rasterio.open(filename, "w",**meta) as dest:
            dest.write(out_image)
            
        if augmentation==True:
            #HORIZONTAL FLIP
            out_image1=numpy.squeeze(out_image)
            out_image1=numpy.fliplr(out_image1)
            out_image1=numpy.expand_dims(out_image1,axis=0)
            meta.update({"driver":Image1.driver,
                         "height":out_image1.shape[1],
                         "width":out_image1.shape[2],
                         "count":out_image1.shape[0],
                         "transform":out_transform})
            filename=os.path.join(Output,'Label/',f"Set_{Dataset}_Im_{number+1}_HorizontalFlip_D{maptype}_Label.tif")
            with rasterio.open(filename, "w",**meta) as dest:
                dest.write(out_image1)

            #VERTICAL FLIP
            out_image2=numpy.squeeze(out_image)
            out_image2=numpy.flipud(out_image2)
            out_image2=numpy.expand_dims(out_image2,axis=0)
            meta.update({"driver":Image1.driver,
                         "height":out_image2.shape[1],
                         "width":out_image2.shape[2],
                         "count":out_image2.shape[0],
                         "transform":out_transform})
            filename=os.path.join(Output,'Label/',f"Set_{Dataset}_Im_{number+1}_VerticalFlip_D{maptype}_Label.tif")
            with rasterio.open(filename, "w",**meta) as dest:
                dest.write(out_image2)   
                
                
            #ROTATION
            out_image4=numpy.squeeze(out_image)
            pilimage=Image.fromarray(out_image4)
            rotated_image=pilimage.rotate(rotation)
            out_image4=numpy.array(rotated_image)
            out_image4=numpy.expand_dims(out_image4,axis=0)                
            meta.update({
                "driver":Image1.driver,
                "height":out_image4.shape[1],
                "width":out_image4.shape[2],
                "count":out_image4.shape[0],
                "transform":out_transform
                })
            filename4=os.path.join(Output,'Label/',f"Set_{Dataset}_Im_{number+1}_Rotated_{rotation}dg_D{maptype}_Label.tif")
            with rasterio.open(filename4, "w",**meta) as dest:
                dest.write(out_image4)
                            
def generate_random_clippings(Dataset,image_path,label_path, output_dir, num_clippings, min_size, max_size,maptype=0):
    """
    Generates random clippings of an image and saves them to the specified directory.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the generated clippings.
        num_clippings (int): Number of random clippings to generate.
        min_size (int): Minimum size (width and height) of the clippings.
        max_size (int): Maximum size (width and height) of the clippings.
    """
    # Open the input imag
    image = rasterio.open(image_path)
    label=rasterio.open(label_path)

    #0=West, 2=East, 1=South and 3=North
    a=label.bounds
    MinLong=a[0]
    MinLat=a[1]
    MaxLong=a[2]
    MaxLat=a[3]
    BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
    
    #Degrees per pixel 
    deg_pp=16**(-1)
  
    # Ensure output directory exists    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_clippings):
        Marker=True
        #print(i)
        random.seed(None)
        while Marker==True:
            #Generate a random center point between BoundingBox's bounds. 
            
            #Now choose a pixel amount from Min and Max size
            long_width=max_size*deg_pp
            lat_width=max_size*deg_pp   
            
            #Generate a random center point between BoundingBox's bounds.
            ClipMinLat=MinLat+lat_width
            ClipMaxLat=MaxLat-lat_width
            ClipMinLong=MinLong+long_width
            ClipMaxLong=MaxLong-long_width
            
            CenterLat=random.uniform(ClipMinLat,ClipMaxLat)
            CenterLon=random.uniform(ClipMinLong,ClipMaxLong)
            #print(CenterLat,CenterLon)
            
            #Then add that to the center points to create a box. 
            MinLong1=CenterLon-long_width
            MinLat1=CenterLat-lat_width
            MaxLong1=CenterLon+long_width
            MaxLat1=CenterLat+lat_width
            AoI=Polygon(((MinLong1,MinLat1),(MaxLong1,MinLat1),(MaxLong1,MaxLat1),(MinLong1,MaxLat1)))
            rotatedeg=random.uniform(1,15)
            #Now check that the box is contained within the original BoundingBox
            if BoundingBox.contains(AoI):
                out_image,out_transform=rasterio.mask.mask(image,[AoI],crop=True)
                SaveImage(Dataset,image_path,out_image,out_transform,OutputDest,i,label=False,augmentation=True,rotation=rotatedeg,maptype=maptype)
                    
                out_label,label_transform=rasterio.mask.mask(label,[AoI],crop=True)
                SaveImage(Dataset,label_path,out_label,label_transform,OutputDest,i,label=True,augmentation=True,rotation=rotatedeg,maptype=maptype)
                #If the image is contained in the lager area, then we make Marker false so it can move on to the next clipping
                Marker=False
                rotatedeg=[]
                print(f"Saved image #{i}")
            else:
                Marker=True #Just in case

#print("Saving training images")
generate_random_clippings(Dataset,GravClip,GravLabel,OutputDest,num_clippings,min_size,max_size,maptype)

    


