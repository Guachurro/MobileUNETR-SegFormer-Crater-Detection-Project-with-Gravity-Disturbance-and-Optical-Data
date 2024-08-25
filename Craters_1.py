# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:41:12 2024

@author: Guach
"""
#Code to read the crater database and create a georeferenced set of polygons from it. 
import geopandas as gpd

import pandas as pd

import rasterio

from shapely import Point, Polygon

from matplotlib import pyplot as plt

from rasterio import mask

from rasterio.windows import Window

import numpy

import pyproj

from pyproj import Geod

import random

#import cartopy.crs as ccrs

import numpy

from PIL import Image

import os

from numpy import nan
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

######Defining the ellipsoid used by the Shapefile, which is the dataset we're doing calculations on.(Not going to be used for now) 
#a=1737.8981*1000 #Equatorial radius
#b=1737.1513*1000 #Polar Radius
#f=(a-b)/a
#e2=(1-b**2)/a**2
#geod=Geod(ellps="Moon_2000_IAU_IAG")

#DEM
#Run this to get all the files in place
#%% 
print('Reading files')
#Craters Shapefile (This was created from the Robbins 2018 Database).
Shape=gpd.read_file("F:\Academics/OSU/Moon Related Projects/Crater Detection Project/Lunar Crater Database/Robbins database USGS 2018/Catalog_Moon_Release_20180815_shapefile180/Catalog_Moon_Release_20180815_1kmPlus_180.shp")
#Proj=gpd.read_file("F:\Academics\OSU\Moon Related Projects\Crater Detection Project\Lunar Crater Database\Robbins database USGS 2018\Catalog_Moon_Release_20180815_shapefile180\Catalog_Moon_Release_20180815_1kmPlus_180.prj")
#Elevation Map
Moon="F:/Academics/OSU/Moon Related Projects/Global/SLDEM2015_256_60S_60N_000_360.JP2"
#Gravity disturbance product
MoonGrav="F:/Academics/OSU/Moon Related Projects/gggrx_1200a_dist_l1200.tif"

#CRS of the DEM. Geographic 2D CRS
CRS_DEM=rasterio.open(MoonGrav).crs
CRS_SH=Shape.crs

#Create a Polygon that will be the Area of Interest to acquire craters
MinLat=-20
MaxLat=20
MinLong=3
MaxLong=23 
AoI=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
#%%
#In this Block we'll be removing all the craters that aren't centered in our desired AoI
list=[]
for index, row in Shape.iterrows():
    if AoI.contains(Shape['geometry'][index]):
        #If its contained you're good. Added value for debugging
        print(Shape['geometry'][index])
    else:#Add the index value to a list so we can drop them all later from the Geodataframe
        list.append(index)

Shape=Shape.drop(list)
print('Finished eliminating undesired rows')
#%% Need to change this block later to use pyproj instead. Must figure out how to designate my own Ellipsoid information, as Pyproj doesn't currently have Moon based ellipsoids.
#####Prepare a polygon for each Crater that will be used to label and extract images later on
print('Creating Crater Polygons')       
#####Crater radius defined as half of the diameter of a circular fit. Original is in kilometers, we need it in meters.  
Shape['Radius']=(Shape['DIAM_C_IM']/2)*1000
#####Project into cylindrical system because Shapely will perform operatons assuming a flat surface. There will be errors? 
print('Projecting into Cylindrical system')
Shape=Shape.to_crs(crs1)
print('Creating crater polygons')
Shape['Boundaries']=Shape['geometry'].buffer(Shape['Radius'])
print('Projecting back into Geographic')
Shape=Shape.to_crs(CRS_SH)
print('Created GDF. Dismissing unwanted data. ')
#####Remove unwanted data from variable
Shape=Shape.drop(columns=Shape.iloc[:,1:17])
print('Creating AoI Polygon')
#%%
#This block will generat ethe images and labels of craters given a GeoDataFrame
print('Moving into loop')
#Obtain all craters in desired area with a radius of 10 kilometers or more
for index, row in Shape.head(5).iterrows():
    with rasterio.open(MoonGrav) as src:
        with rasterio.open(Moon) as src1:
            ####Boundaries of the polygon containing the crater
            MinLat=row['Boundaries'].bounds[1]
            MaxLat=row['Boundaries'].bounds[3]
            MinLong=row['Boundaries'].bounds[0]
            MaxLong=row['Boundaries'].bounds[2]
            
            ####Lat/Long Offset by calculating their total extent and dividing it by 2. That number is then used to shift the window by applying a random percentage from 1 to 100
            # (Max - Min)/2 gets us half of the total window span in degrees
            # (random.randint(1,100)/100) Generates a random Integer from 1 to 100 that is turned into a percentage. This is multiplies by the window span to know how much we're moving from target center.
            # ((-1)**random.randint(1,100)) Will determine a positive or negative value that will determine whether the shift is positive or negative
            LatOffset=((MaxLat-MinLat)/2)*(random.randint(1,100)/100)*((-1)**random.randint(1,2))
            LonOffset=((MaxLong-MinLong)/2)*(random.randint(1,100)/100)((-1)**random.randint(1,2))
                
            # Min and Max values are both offset by the same ammount to maintain the same original window size. 
            Minlat=MinLat+LatOffset
            MaxLat=MaxLat+LatOffset
            MinLong=MinLong+LonOffset
            MaxLong=MaxLong+LonOffset
            
            ####Create a bounding box that has been offset from the Crater in some direction 
            BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
            
            ####Making sure this is empty just in case
            im=[]
            ####This output is the cropping of te Gravity Data
            out_image, out_transform = rasterio.mask.mask(src,[BoundingBox], crop=True)
            ####This output is the cropping of the Lunar DEM (SLDEM2015)
            out_image1, out_transform1 = rasterio.mask.mask(src1,[BoundingBox], crop=True)
            ####Remove first dimension and make Nodata values equal to 0
            im=numpy.squeeze(out_image)
            im[im==-32767]=0
            im1=numpy.squeeze(out_image1)
            im1[im1==-32767]=0
            
            #Create plot GRAVITY
            plt.imshow(im, cmap='viridis')
            plt.title(row['CRATER_ID'])
            #Show plot
            plt.show()

            #Create plot Crater DEM (Meant to serve as a debugging feature)
            plt.imshow(im1, cmap='gist_gray')
            plt.title(row['CRATER_ID'])
            #Show plot
            plt.show()                    
            
            #The next step is to recreate each Gravity Map cropping and properly label it. 
                    #Save as GeoTiff
                    #output_dir="F:/Academics/OSU/Moon Related Projects/CraterImages"
                    #filename=os.path.join(output_dir,f"{row['CRATER_ID']}.tif")
                    #with rasterio.open(filename, "w",**src.profile) as dest:
                     #   dest.write(out_image)
print('Exit loop')
