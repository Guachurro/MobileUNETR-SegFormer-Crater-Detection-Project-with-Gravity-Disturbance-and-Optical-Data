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

#Pixel Per Degree
ppd=128
#Kilometer per Pixel
kmpx=0.2369011752
#Pixel Per Kilometer
pxkm=kmpx**-1
#Kilometer per Degree
kmd=ppd*kmpx
#Degrees per Kilometer
dkm=kmd**-1
#DEM
#Run this to get all the files in place
#%% 
print('Reading files')
Moon=rasterio.open("D:/Academics/OSU/Moon Related Projects/Global/SLDEM2015_128_60S_60N_000_360.JP2")
#Gravity disturbance
MoonGrav=rasterio.open("D:/Academics/OSU/Moon Related Projects/gggrx_1200a_dist_l1200.tif")
#PixelsPerDegree (For image data)
gppd=16
#kilometers per pixel(For image data) You need at least 2 kilometers
gkmpix=1895/1000
#Kilometers per degree
gkmdeg=gppd*gkmpix
#Degrees per kilometer
gdkm=gkmdeg**(-1)

#Crater Database
Robbins=pd.read_excel('D:/Academics/OSU/Moon Related Projects/Crater Detection Project/Lunar Crater Database/Robbins database USGS 2018/Lunar Crater Database Robbins 2018.xlsx')
print('Finshed files. Extracting WKT')
#DEM Well Known Text
WKT=Moon.crs
WKT1=MoonGrav.crs

#%%
print('Creating GDF and buffer zones')
#Lon circi IMG has range 0 to 360. Alter this to -180 to 180 by subtracting 360 from anything larger than 180*
#Subtract 360 from every Longitude value greater than 180
Longe=[]
for index, row in Robbins.iterrows():
    if row['LON_CIRC_IMG']>180:
        Longe.append(row['LON_CIRC_IMG']-360)
    else:
        Longe.append(row['LON_CIRC_IMG'])
#Create GeoDataFrame        
Craters=gpd.GeoDataFrame(Robbins, geometry= gpd.points_from_xy(Longe-(Robbins['DIAM_CIRC_IMG']/2)*dkm,Robbins['LAT_CIRC_IMG']-(Robbins['DIAM_CIRC_IMG']/2)*dkm), crs=WKT)
#Create Buffer zones around crater centers. Radius is expected in degrees. 
Craters['Shapes'] = Craters.buffer((Robbins['DIAM_CIRC_IMG']/2)*dkm)
Craters['Radius']=Robbins['DIAM_CIRC_IMG']/2
#Remove unwanted data from variable
Craters=Craters.drop(['LAT_CIRC_IMG',	'LON_CIRC_IMG',	'LAT_ELLI_IMG',	'LON_ELLI_IMG',	'DIAM_CIRC_IMG',	'DIAM_CIRC_SD_IMG',	'DIAM_ELLI_MAJOR_IMG',	'DIAM_ELLI_MINOR_IMG',	'DIAM_ELLI_ECCEN_IMG',	'DIAM_ELLI_ELLIP_IMG',	'DIAM_ELLI_ANGLE_IMG',	'LAT_ELLI_SD_IMG',	'LON_ELLI_SD_IMG',	'DIAM_ELLI_MAJOR_SD_IMG',	'DIAM_ELLI_MINOR_SD_IMG',	'DIAM_ELLI_ANGLE_SD_IMG',	'DIAM_ELLI_ECCEN_SD_IMG',	'DIAM_ELLI_ELLIP_SD_IMG',	'ARC_IMG',	'PTS_RIM_IMG'],axis=1)
print('Defining bounding region')
#Limiting area from where craters are obtained. 
CenterPoint=(24,-144)
Area=20
MinLat=CenterPoint[0]-Area
MaxLat=CenterPoint[0]+Area
MinLong=CenterPoint[1]-Area
MaxLong=CenterPoint[1]+Area
print('Creating Polygon')
BoundingBox=Polygon(((MinLong,MinLat),(MaxLong,MinLat),(MaxLong,MaxLat),(MinLong,MaxLat)))

#%%
print('Moving into loop')
#Append these for every crater you find inside, or outside of the given region
Coordinates=[]
Outside=[]
#Obtain all craters in desired area with an area of 5 kilometers or more
for index, row in Craters.head(550).iterrows():
    with rasterio.open("D:/Academics/OSU/Moon Related Projects/gggrx_1200a_dist_l1200.tif") as src:
        if row['Radius']>10:
            #if BoundingBox.contains(row['Shapes'])==True:
                    #Making sure this is empty just in case
                    im=[]
                    out_image, out_transform = rasterio.mask.mask(src,[row['Shapes']], crop=True)
                    #Append the coordinate and index inside polygon
                    Coordinates.append([index,row['CRATER_ID'],row['geometry']])
                    #Remove first dimension
                    im=numpy.squeeze(out_image)
                    #Create plot
                    plt.imshow(im, cmap='viridis')
                    plt.title(row['CRATER_ID'])
                    #Show plot
                    plt.show()
                    #Save as GeoTiff
                    out_meta=src.meta
                    profile=src.profile
                    profile["height"]=out_image.shape[1]
                    profile["width"]=out_image.shape[2]
                    profile["transform"]=out_transform
                    #with rasterio.open(row['CRATER_ID'],"w",**profile) as dest:
                        #dest.write(out_image)
            #else:
                    Outside.append([index,row['CRATER_ID'], row['geometry']])
print('Exit loop')