# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:41:12 2024

@author: Guach
"""
#Necessary modules and file destinations
import geopandas as gpd

import pandas as pd

import rasterio

from shapely import Point, Polygon

from matplotlib import pyplot as plt

from rasterio import mask

from rasterio.windows import Window

import numpy

#import pyproj

#from pyproj import Geod

import random

from PIL import Image

import os

from numpy import nan


def interpolate_pixel(data, mask):
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            if mask[i, j]:
                neighbors = data[i-1:i+2, j-1:j+2]
                data[i, j] = numpy.mean(neighbors[neighbors != -32676])
        return data

#Gravity Map Resolution Information
#PixelsPerDegree (For image data)
ppd=16
#kilometers per pixel(For image data) You need at least 2 kilometers
kmpix=1.895
#Kilometers per degree
kmdeg=ppd*kmpix
#Degrees per kilometer
dkm=kmdeg**(-1)

#Moon2000 Equidistant Cylindrical (Uses meters as the units)
crs1="Moon_2000_Equidistant_Cylindrical"

###SLDEM2015
Moon="F:/Academics/OSU/Thesis Documents/Lunar DEMs/SLDEM2015_256_60S_60N_000_360.JP2"

###Gravity Map GGGrx1200a
MoonGrav="F:/Academics/OSU/Thesis Documents/Lunar Gravity Maps/gggrx_1200a_dist_l1200.tif"

###Output Place for file generation
#Root folder where I will store images and labels 
Output="F:/Academics/OSU/Thesis Documents/Images and Labels/"
#This is the label image
GravClip="F:/Academics/OSU/Thesis Documents/Images and Labels/0N_13E_5kR_Craters.tif"


#Run this to get all the files in place
print('Reading files')
####Craters Shapefile (This was created from the Robbins 2018 Database).
#If k=1 Then the Shapefile has already been edited to the size we want so we read the smaller version
k=1
if k==1:
    Shape=gpd.read_file("F:/Academics/OSU/Thesis Documents/CratersAOI/CratersAOI.shp")
else: 
    Shape=gpd.read_file("F:/Academics/OSU/Thesis Documents/Crater Databases/Robbins database USGS 2018/Catalog_Moon_Release_20180815_shapefile180/Catalog_Moon_Release_20180815_1kmPlus_180.shp")

#CRS of the DEM. Geographic 2D CRS
CRS_DEM=rasterio.open(MoonGrav).crs
CRS_SH=Shape.crs

#Create a Polygon that will be the Area of Interest to acquire craters
print('Create AOI Box')
MinLat=-20
MaxLat=20
MinLong=3
MaxLong=23 
AoI=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))


#Eliminate non AOI Craters and create radius
#Skip when k=0
if k==1:
    list=[]
    for index, row in Shape.iterrows():
        if AoI.contains(Shape['geometry'][index]) :
            #If its contained you're good. Added value for debugging
            k='nothing'
        else:#Add the index value to a list so we can drop them all later from the Geodataframe
            list.append(index)

    Shape=Shape.drop(list)
    list=[]
    print('Eliminated Craters outside of AOI')
    print('Creating Crater Radius')       
    #####Crater radius defined as half of the diameter of a circular fit. Original is in kilometers, we need it in meters.  
    Shape['Radius']=(Shape['DIAM_C_IM']/2)*1000
    print('Eliminating unneccessary columns')
    Shape=Shape.drop(columns=Shape.iloc[:,1:17])
    
    
#Need to change this block later to use pyproj instead. Must figure out how to designate my own Ellipsoid information, as Pyproj doesn't currently have Moon based ellipsoids.
#Likewise, if we have already created the edited database we dont need to repeat these steps. We skip the whole block. 
#####Prepare a polygon for each Crater that will be used to label and extract images later on
###Eliminate craters with less than 5k meter radius (10 kilometers)
list=[]
print('Preparing list of craters')
for index, row in Shape.iterrows():
    if Shape['Radius'][index]<5000:
        list.append(index)
Shape=Shape.drop(list)
print('Eliminated Craters with radius criteria')
#####Project into cylindrical system because Shapely will perform operatons assuming a flat surface. There will be errors? 
print('Projecting into Cylindrical system')
Shape=Shape.to_crs(crs1)
print('Creating crater polygons')
Shape['Boundaries']=Shape['geometry'].buffer(Shape['Radius'])
print('Projecting back into Geographic')
Shape=Shape.to_crs(CRS_SH)
Shape['Boundaries']=Shape['Boundaries'].to_crs(CRS_SH)

print('Preparing to clip Gravity Map')
TIF=rasterio.open(MoonGrav)
####Create the mask using only the chosen craters. 
out_image, out_transform = rasterio.mask.mask(TIF, Shape['Boundaries'], crop=True)
meta=TIF.profile
meta.update({"driver":TIF.driver,
             "height":out_image.shape[1],
             "width":out_image.shape[2],
             "count":out_image.shape[0],
             "transform":out_transform})
#%%
print('Saving Clip')
#Save as GeoTiff
output_dir=Output
filename=os.path.join(Output+"0N_13E_5kR_Craters.tif")
with rasterio.open(filename, "w",**meta) as dest:
    dest.write(out_image)

#%%
#This block will generat ethe images and labels of craters given a GeoDataFrame without performing inflation techniques. 
print('Moving into loop without inflatiion')
Additional='_Bare'
#Obtain all craters in desired area with a radius of 10 kilometers or more
#Take a centered image first. 
for index, row in Shape.iterrows():
    with rasterio.open(MoonGrav) as src:
        with rasterio.open(GravClip) as src1:
            ####Boundaries of the polygon containing the crater
            MinLat=row['Boundaries'].bounds[1]
            MaxLat=row['Boundaries'].bounds[3]
            MinLong=row['Boundaries'].bounds[0]
            MaxLong=row['Boundaries'].bounds[2]
            CraterID=row['CRATER_ID']+Additional

            ####Create a bounding box that has been offset from the Crater in some direction. 
            BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
            
            ####This output is the cropping of the Gravity Data.
            out_image, out_transform = rasterio.mask.mask(src,[BoundingBox], crop=True)
            out_image[out_image==-32767]=0
            meta=src.profile
            meta.update({"driver":src.driver,
                         "height":out_image.shape[1],
                         "width":out_image.shape[2],
                         "count":out_image.shape[0],
                         "transform":out_transform})
            
            
            filename=os.path.join(Output+'Image/'+CraterID+'.tif')
            with rasterio.open(filename, "w",**meta) as dest:
                dest.write(out_image)
            
            
            ####This output is the cropping of the Label. 
            out_image1, out_transform1 = rasterio.mask.mask(src1,[BoundingBox], crop=True)
            #out_image1= numpy.expand_dims(out_image1, axis=-1)
            out_image1[out_image1!=-32767]=1
            out_image1[out_image1==-32767]=0
            meta1=src1.profile
            meta1.update({"driver":src1.driver,
                        "height":out_image1.shape[1],
                        "width":out_image1.shape[2],
                        "count":out_image1.shape[0],
                        "transform":out_transform1})

            #out_image1 = numpy.expand_dims(out_image1, axis=-1)
            filename=os.path.join(Output+'Label/'+CraterID+'_Label'+'.tif')
            with rasterio.open(filename, "w",**meta1) as dest:
                dest.write(out_image1)
                
            #print('Crater done')
print('Exit loop')


#%%
#This block will generat ethe images and labels of craters given a GeoDataFrame without performing inflation techniques. 
print('Moving into loop with translation')
Additional='_Translation'
#Obtain all craters in desired area with a radius of 10 kilometers or more
#Take a centered image first. 
for index, row in Shape.iterrows():
    with rasterio.open(MoonGrav) as src:
        with rasterio.open(GravClip) as src1:
            ####Boundaries of the polygon containing the crater
            MinLat=row['Boundaries'].bounds[1]
            MaxLat=row['Boundaries'].bounds[3]
            MinLong=row['Boundaries'].bounds[0]
            MaxLong=row['Boundaries'].bounds[2]
            CraterID=row['CRATER_ID']+Additional
            
            ####Lat/Long Offset by calculating their total extent and dividing it by 2. That number is then used to shift the window by applying a random percentage from 1 to 100
            # (Max - Min)/2 gets us half of the total window span in degrees
            # (random.randint(1,100)/100) Generates a random Integer from 1 to 100 that is turned into a percentage. This is multiplies by the window span to know how much we're moving from target center.
            # ((-1)**random.randint(1,100)) Will determine a positive or negative value that will determine whether the shift is positive or negative
            LatOffset=((MaxLat-MinLat)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
            LonOffset=((MaxLong-MinLong)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
                
            
            # Min and Max values are both offset by the same ammount to maintain the same original window size. 
            Minlat=MinLat+LatOffset
            MaxLat=MaxLat+LatOffset
            MinLong=MinLong+LonOffset
            MaxLong=MaxLong+LonOffset
            
            ####Create a bounding box that has been offset from the Crater in some direction. 
            BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
            
            ####This output is the cropping of te Gravity Data.
            out_image, out_transform = rasterio.mask.mask(src,[BoundingBox], crop=True)
            out_image[out_image==-32767]=0
            meta=src.profile
            meta.update({"driver":src.driver,
                         "height":out_image.shape[1],
                         "width":out_image.shape[2],
                         "count":out_image.shape[0],
                         "transform":out_transform})
            filename=os.path.join(Output+'Image/'+CraterID+'.tif')
            with rasterio.open(filename, "w",**meta) as dest:
                dest.write(out_image)
            
            
            ####This output is the cropping of the Label. 
            out_image1, out_transform1 = rasterio.mask.mask(src1,[BoundingBox], crop=True)
            #out_image1= numpy.expand_dims(out_image1, axis=-1)
            out_image1[out_image1!=-32767]=1
            out_image1[out_image1==-32767]=0
            meta1=src1.profile
            meta1.update({"driver":src1.driver,
                        "height":out_image1.shape[1],
                        "width":out_image1.shape[2],
                        "count":out_image1.shape[0],
                        "transform":out_transform1})

            filename=os.path.join(Output+'Label/'+CraterID+'_Label'+'.tif')
            with rasterio.open(filename, "w",**meta1) as dest:
                dest.write(out_image1)
                
            #print('Crater done')
print('Exit loop')

#%%
#This block will generat ethe images and labels of craters given a GeoDataFrame without performing inflation techniques. 
print('Moving into loop with translation and horizontal flip')
Additional='_Translation_FlipH'
#Obtain all craters in desired area with a radius of 10 kilometers or more
#Take a centered image first. 
for index, row in Shape.iterrows():
    with rasterio.open(MoonGrav) as src:
        with rasterio.open(GravClip) as src1:
            ####Boundaries of the polygon containing the crater
            MinLat=row['Boundaries'].bounds[1]
            MaxLat=row['Boundaries'].bounds[3]
            MinLong=row['Boundaries'].bounds[0]
            MaxLong=row['Boundaries'].bounds[2]
            CraterID=row['CRATER_ID']+Additional
            
            ####Lat/Long Offset by calculating their total extent and dividing it by 2. That number is then used to shift the window by applying a random percentage from 1 to 100
            # (Max - Min)/2 gets us half of the total window span in degrees
            # (random.randint(1,100)/100) Generates a random Integer from 1 to 100 that is turned into a percentage. This is multiplies by the window span to know how much we're moving from target center.
            # ((-1)**random.randint(1,100)) Will determine a positive or negative value that will determine whether the shift is positive or negative
            LatOffset=((MaxLat-MinLat)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
            LonOffset=((MaxLong-MinLong)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
                
            
            # Min and Max values are both offset by the same ammount to maintain the same original window size. 
            Minlat=MinLat+LatOffset
            MaxLat=MaxLat+LatOffset
            MinLong=MinLong+LonOffset
            MaxLong=MaxLong+LonOffset
            
            ####Create a bounding box that has been offset from the Crater in some direction. 
            BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
            
            ####This output is the cropping of te Gravity Data.
            out_image, out_transform = rasterio.mask.mask(src,[BoundingBox], crop=True)
            out_image[out_image==-32767]=0
            out_image=numpy.fliplr(out_image)
            meta=src.profile
            meta.update({"driver":src.driver,
                         "height":out_image.shape[1],
                         "width":out_image.shape[2],
                         "count":out_image.shape[0],
                         "transform":out_transform})
            filename=os.path.join(Output+'Image/'+CraterID+'.tif')
            with rasterio.open(filename, "w",**meta) as dest:
                dest.write(out_image)
            
            
            ####This output is the cropping of the Label. 
            out_image1, out_transform1 = rasterio.mask.mask(src1,[BoundingBox], crop=True)

            out_image1=numpy.fliplr(out_image1)
            out_image1[out_image1!=-32767]=1
            out_image1[out_image1==-32767]=0
            meta1=src1.profile
            meta1.update({"driver":src1.driver,
                        "height":out_image1.shape[1],
                        "width":out_image1.shape[2],
                        "count":out_image1.shape[0],
                        "transform":out_transform1})

            filename=os.path.join(Output+'Label/'+CraterID+'_Label'+'.tif')
            with rasterio.open(filename, "w",**meta1) as dest:
                dest.write(out_image1)
                
            #print('Crater done')
print('Exit loop')
#%%
#This block will generat ethe images and labels of craters given a GeoDataFrame without performing inflation techniques. 
print('Moving into loop with Gaussian Noise')
Additional='_Translation_GNoise'
#Obtain all craters in desired area with a radius of 10 kilometers or more
#Take a centered image first. 
for index, row in Shape.iterrows():
    with rasterio.open(MoonGrav) as src:
        with rasterio.open(GravClip) as src1:
            ####Boundaries of the polygon containing the crater
            MinLat=row['Boundaries'].bounds[1]
            MaxLat=row['Boundaries'].bounds[3]
            MinLong=row['Boundaries'].bounds[0]
            MaxLong=row['Boundaries'].bounds[2]
            CraterID=row['CRATER_ID']+Additional
            numpy.random.seed(30)
            
            
            ####Lat/Long Offset by calculating their total extent and dividing it by 2. That number is then used to shift the window by applying a random percentage from 1 to 100
            # (Max - Min)/2 gets us half of the total window span in degrees
            # (random.randint(1,100)/100) Generates a random Integer from 1 to 100 that is turned into a percentage. This is multiplies by the window span to know how much we're moving from target center.
            # ((-1)**random.randint(1,100)) Will determine a positive or negative value that will determine whether the shift is positive or negative
            LatOffset=((MaxLat-MinLat)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
            LonOffset=((MaxLong-MinLong)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
                
            
            # Min and Max values are both offset by the same ammount to maintain the same original window size. 
            Minlat=MinLat+LatOffset
            MaxLat=MaxLat+LatOffset
            MinLong=MinLong+LonOffset
            MaxLong=MaxLong+LonOffset
            
            ####Create a bounding box that has been offset from the Crater in some direction. 
            BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
            
            ####This output is the cropping of te Gravity Data.
            out_image, out_transform = rasterio.mask.mask(src,[BoundingBox], crop=True)
            out_image[out_image==-32767]=0
            #Prepare noise for Data
            Noise=numpy.random.normal(loc=0,scale=1,size=(out_image.shape[0],out_image.shape[1],out_image.shape[2]))

            out_image=out_image+Noise
            meta=src.profile
            meta.update({"driver":src.driver,
                         "height":out_image.shape[1],
                         "width":out_image.shape[2],
                         "count":out_image.shape[0],
                         "transform":out_transform})
            filename=os.path.join(Output+'Image/'+CraterID+'.tif')
            with rasterio.open(filename, "w",**meta) as dest:
                dest.write(out_image)
            
            
            ####This output is the cropping of the Label. The label also doesn't require any noise or alterations so long as you didn't flip the dataset. 
            out_image1, out_transform1 = rasterio.mask.mask(src1,[BoundingBox], crop=True)
            #out_image1= numpy.expand_dims(out_image1, axis=-1)
            out_image1[out_image1!=-32767]=1
            out_image1[out_image1==-32767]=0
            meta1=src1.profile
            meta1.update({"driver":src1.driver,
                        "height":out_image1.shape[1],
                        "width":out_image1.shape[2],
                        "count":out_image1.shape[0],
                        "transform":out_transform1})

            filename=os.path.join(Output+'Label/'+CraterID+'_Label'+'.tif')
            with rasterio.open(filename, "w",**meta1) as dest:
                dest.write(out_image1)
                
            #print('Crater done')
print('Exit loop')
