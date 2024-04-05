# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:41:12 2024

@author: Guach
"""
#Code to read the crater database and create a georeferenced set of polygons from it. 
import geopandas as gpd

import pandas as pd

import rasterio

from shapely import Polygon

from matplotlib import pyplot as plt

from rasterio import mask

from rasterio.windows import Window

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

#DEM
#Run this to get all the files in place
#%% 
print('Reading files')
#Craters Shapefile
Shape=gpd.read_file("F:\Academics/OSU/Moon Related Projects/Crater Detection Project/Lunar Crater Database/Robbins database USGS 2018/Catalog_Moon_Release_20180815_shapefile180/Catalog_Moon_Release_20180815_1kmPlus_180.shp")
#Elevation Map
Moon="F:/Academics/OSU/Moon Related Projects/Global/SLDEM2015_256_60S_60N_000_360.JP2"
#Gravity disturbance
MoonGrav="F:/Academics/OSU/Moon Related Projects/gggrx_1200a_dist_l1200.tif"
print('Finshed files. Extracting WKT')

#DEM Well Known Text
WKT=rasterio.open(MoonGrav).crs

#%%
print('Creating GDF and buffer zones')       
#Populate existing Data Frame with Radius(in degrees) and Bounding polygon for each crater
Shape['Radius']=Shape['DIAM_C_IM']/2
Shape['Shapes']=Shape['geometry'].buffer(Shape['Radius'])
#Remove unwanted data from variable
Shape=Shape.drop(columns=Shape.iloc[:,1:17])

#%%
print('Moving into loop')
#Obtain all craters in desired area with a radius of 10 kilometers or more
for index, row in Shape.head(5).iterrows():
    with rasterio.open(MoonGrav) as src:
        with rasterio.open(Moon) as src1:
            #if row['Radius']>10:
                #Making sure this is empty just in case
                im=[]
                out_image, out_transform = rasterio.mask.mask(src,[row['Shapes']], crop=True)
                MinLat=row['Shapes'].bounds[1]
                MaxLat=row['Shapes'].bounds[3]
                MinLong=row['Shapes'].bounds[0]
                MaxLong=row['Shapes'].bounds[2]
                BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))
                out_image1, out_transform1 = rasterio.mask.mask(src1,[BoundingBox], crop=True)
                #Remove first dimension
                im=numpy.squeeze(out_image)
                im[im==-32767]=0
                im1=numpy.squeeze(out_image1)
                im1[im1==-32767]=0
                #Create plot GRAVITY
                plt.imshow(im, cmap='viridis')
                plt.title(row['CRATER_ID'])
                #Show plot
                plt.show()

                #Create plot Crater
                plt.imshow(im1, cmap='gist_gray')
                plt.title(row['CRATER_ID'])
                #Show plot
                plt.show()                    
                    #Save as GeoTiff
                    #output_dir="F:/Academics/OSU/Moon Related Projects/CraterImages"
                    #filename=os.path.join(output_dir,f"{row['CRATER_ID']}.tif")
                    #with rasterio.open(filename, "w",**src.profile) as dest:
                     #   dest.write(out_image)
print('Exit loop')