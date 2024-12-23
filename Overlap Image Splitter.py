# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:13:25 2024

@author: Guach
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
import albumentations as A


def split_geotiff(image_path,label_path, output_folder, patch_size, overlap,augmentation=False):
    # Open the GeoTIFF file
    src=rasterio.open(image_path)
    src1=rasterio.open(label_path)
    # Get image dimensions (number of rows and columns)
    img_width = src.width
    img_height = src.height
    transform = src.transform
        
    img_width1 = src1.width
    img_height1 = src1.height
    transform1 = src1.transform
    
      # Calculate the step sze (patch size minus overlap)
    step_size = int(patch_size * (1 - overlap))
    count=0
    
    #A.RandomSizedCrop(min_max_height=(patch_size//2, patch_size), height=patch_size, width=patch_size, p=0.3),

    
    # Iterate over the image, extracting patches
    for row in range(0, img_height - patch_size + 1, step_size):
        for col in range(0, img_width - patch_size + 1, step_size):
            # Define the window (patch) to extract
            window = Window(col_off=col, row_off=row, width=patch_size, height=patch_size)
            
            # Read the patch data
            patch = src.read(window=window)
            patch1=src1.read(window=window)
            count=count+1
            print(f"Created patch: {count}")
            # Create output filename
            outimage=os.path.join(output_folder,'Image/')
            output_fileimage = os.path.join(outimage,f"patch#{count}_{row}_{col}.tif")
            
            outlabel=os.path.join(output_folder,'Label/')
            output_filelabel = os.path.join(outlabel,f"patch#{count}_{row}_{col}_Label.tif")

            # Create metadata for Image
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'height': patch.shape[1],
                'width': patch.shape[2],
                'transform': rasterio.windows.transform(window, transform)
            })
                
            # Write the patch to a new GeoTIFF file
            with rasterio.open(output_fileimage, 'w', **meta) as dst:
                dst.write(patch)
                    
            # Create metadata for Label
            meta1 = src1.meta.copy()
            meta1.update({
                'driver': 'GTiff',
                'height': patch1.shape[1],
                'width': patch1.shape[2],
                'transform': rasterio.windows.transform(window, transform1)
            })
                    
            # Write the patch to a new GeoTIFF file
            with rasterio.open(output_filelabel, 'w', **meta1) as dst:
                dst.write(patch1)
            
            if augmentation==0:
                #No augmentation
                nothing=[]
            
            elif augmentation==1:
                augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ])
                
                #FIRST AUGMENTATION
                patch_augmented = patch.transpose(1, 2, 0)
                patch1_augmented = patch1.transpose(1, 2, 0)
                    
                augmented = augmentation_pipeline(image=patch_augmented, mask=patch1_augmented)
                patch = augmented['image'].transpose(2, 0, 1)  # Back to CxHxW
                patch1 = augmented['mask'].transpose(2, 0, 1)  # Back to CxHxW
                
                # Create metadata for Image
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'height': patch.shape[1],
                    'width': patch.shape[2],
                    'transform': rasterio.windows.transform(window, transform)
                    })
            
                output_fileimage = os.path.join(outimage,f"{augmentation}_patch#{count}_{row}_{col}_Transform.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_fileimage, 'w', **meta) as dst:
                    dst.write(patch)
                    
                # Create metadata for Label
                meta1 = src1.meta.copy()
                meta1.update({
                    'driver': 'GTiff',
                    'height': patch1.shape[1],
                    'width': patch1.shape[2],
                    'transform': rasterio.windows.transform(window, transform1)})
                    
                output_filelabel = os.path.join(outlabel,f"{augmentation}_patch#{count}_{row}_{col}_Transform_Label.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_filelabel, 'w', **meta1) as dst:
                    dst.write(patch1)
            
            elif augmentation==2:
                augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30,p=0.8),
                ])
                
                #FIRST AUGMENTATION
                patch_augmented = patch.transpose(1, 2, 0)
                patch1_augmented = patch1.transpose(1, 2, 0)
                    
                augmented = augmentation_pipeline(image=patch_augmented, mask=patch1_augmented)
                patch = augmented['image'].transpose(2, 0, 1)  # Back to CxHxW
                patch1 = augmented['mask'].transpose(2, 0, 1)  # Back to CxHxW
                
                # Create metadata for Image
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'height': patch.shape[1],
                    'width': patch.shape[2],
                    'transform': rasterio.windows.transform(window, transform)
                    })
            
                output_fileimage = os.path.join(outimage,f"{augmentation}_patch#{count}_{row}_{col}_Transform.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_fileimage, 'w', **meta) as dst:
                    dst.write(patch)
                    
                # Create metadata for Label
                meta1 = src1.meta.copy()
                meta1.update({
                    'driver': 'GTiff',
                    'height': patch1.shape[1],
                    'width': patch1.shape[2],
                    'transform': rasterio.windows.transform(window, transform1)})
                    
                output_filelabel = os.path.join(outlabel,f"{augmentation}_patch#{count}_{row}_{col}_Transform_Label.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_filelabel, 'w', **meta1) as dst:
                    dst.write(patch1)
            
            
            elif augmentation==3:
                augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45,p=0.5),
                A.CoarseDropout(max_holes=4, max_height=10, max_width=10, p=0.2)  # Random erasing-like effect
            ])
                augmentation_pipeline1 = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=70,p=0.5),
                A.CoarseDropout(max_holes=4, max_height=10, max_width=10, p=0.5)  # Random erasing-like effect
            ])
                
                #FIRST AUGMENTATION
                patch_augmented = patch.transpose(1, 2, 0)
                patch1_augmented = patch1.transpose(1, 2, 0)
                    
                augmented = augmentation_pipeline(image=patch_augmented, mask=patch1_augmented)
                patch = augmented['image'].transpose(2, 0, 1)  # Back to CxHxW
                patch1 = augmented['mask'].transpose(2, 0, 1)  # Back to CxHxW
                
                
                
                # Create metadata for Image
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'height': patch.shape[1],
                    'width': patch.shape[2],
                    'transform': rasterio.windows.transform(window, transform)
                    })
            
                output_fileimage = os.path.join(outimage,f"{augmentation}_patch#{count}_{row}_{col}_Transform.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_fileimage, 'w', **meta) as dst:
                    dst.write(patch)
                    
                # Create metadata for Label
                meta1 = src1.meta.copy()
                meta1.update({
                    'driver': 'GTiff',
                    'height': patch1.shape[1],
                    'width': patch1.shape[2],
                    'transform': rasterio.windows.transform(window, transform1)})
                    
                output_filelabel = os.path.join(outlabel,f"{augmentation}_patch#{count}_{row}_{col}_Transform_Label.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_filelabel, 'w', **meta1) as dst:
                    dst.write(patch1)
                    
                #SECOND AUGMENTATION
                patch_augmented = patch.transpose(1, 2, 0)
                patch1_augmented = patch1.transpose(1, 2, 0)
                
                augmented = augmentation_pipeline1(image=patch_augmented, mask=patch1_augmented)
                patch = augmented['image'].transpose(2, 0, 1)  # Back to CxHxW
                patch1 = augmented['mask'].transpose(2, 0, 1)  # Back to CxHxW
                    
                    
                    
                # Create metadata for Image
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'height': patch.shape[1],
                    'width': patch.shape[2],
                    'transform': rasterio.windows.transform(window, transform)
                    })
                    
                output_fileimage = os.path.join(outimage,f"{augmentation}_patch#{count}_{row}_{col}_Transform1.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_fileimage, 'w', **meta) as dst:
                    dst.write(patch)
                
                # Create metadata for Label
                meta1 = src1.meta.copy()
                meta1.update({
                    'driver': 'GTiff',
                    'height': patch1.shape[1],
                    'width': patch1.shape[2],
                    'transform': rasterio.windows.transform(window, transform1)})
            
                output_filelabel = os.path.join(outlabel,f"{augmentation}_patch#{count}_{row}_{col}_Transform1_Label.tif")
                # Write the patch to a new GeoTIFF file
                with rasterio.open(output_filelabel, 'w', **meta1) as dst:
                    dst.write(patch1)
                

if __name__ == "__main__":
    # Path to the input GeoTIFF file
    train_image_path ="F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/Training/15E_0N_130by70Area_660_Training.tif"
    val_image_path="F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/Validation/65W_0N_30by70Area_660_Validation.tif"
    test_image_path="F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/Test/90E_0N_20by70Area_Test_660.tif"
    # Folder where the patches will be saved
    train_output_folder = "F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/Training/"
    val_output_folder = "F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/Validation/"
    test_output_folder = "F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/Test/"
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(val_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)
    
    # Parameters for patch size and overlap
    patch_size =100  # For example, 512x512 pixels
    overlap = .1  # 30% overlap

    # Call the function to split the GeoTIFF into patches
    #Train
    split_geotiff(train_image_path,os.path.join(train_image_path[:-4]+'_Label.tif'), train_output_folder, patch_size=100, overlap=.05,augmentation=3)
    #Validation
    split_geotiff(val_image_path, os.path.join(val_image_path[:-4]+'_Label.tif'), val_output_folder, patch_size=100, overlap=.3,augmentation=2)
    #Testing
    split_geotiff(test_image_path, os.path.join(test_image_path[:-4]+'_Label.tif'), test_output_folder, patch_size=100, overlap=.3,augmentation=2)
    
    #Augmentation Method 0 = No Augmentation performed
    #Augmentation Method 1 = Low, done once. Image flips
    #Augmentation Method 2 = Mild, done once. H/V Image flips and rotation
    #Augmentation Method 3 = Aggresive, done twice. H/V Flips,Rotation and image erasures.
    
    