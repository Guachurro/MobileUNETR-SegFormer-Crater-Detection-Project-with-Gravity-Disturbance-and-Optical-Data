The contents of this repository are as follows: 
1) 10to200_10E_0N_180by70_Area_660_Label.tif
      - Regional label dpicting craters between 12km to 200km in diameter
      - Not split into Training/Validation/Testing. There is a separate zip file for this.
3) 12to200_10E_0N_180by70Area_BoundariesOnly.7z
      - Zip file containing shapefile boundaries for the craters depicted in the Regional Label
      - They are not split into Training/Validation/testing regions. There is a seperate zip file for this.
4) Area of Interest Data Types Normalized (Zip file with 3 Normalized regional Images. All normlized, and formatted to have 3 channels)
      - These are the complete region, and are not split into Training, Validation and test areas. 
5) Regional Image Splits
      - Train_Val_Test_Optical_Grav660.7z (Regional Image split into 3 regions. Gravity+Optical Fusion)
      - Train_Val_Test_Grav660.7z (Regional Image split into 3 regions. Gravity Disturbance only))
      - Train_Val_Test_Optical.7z (Regional Image split into 3 regions. Optical only)
      - Train_Val_Test_Optical_Label.7z (Regional label split into 3 regions. Use the same labels for each data Type)
6) requirements.txt
      - Python modules used
7) USGS Job Information.log
      - File containing details for the acquisition of the Global Optical Mosaic used in this work. In case you wish to obtain a replica of it. 
8) Overlap Image Splitter.py
      - This code will take the Training/Validation/Test portions of the Image and Label and splt them into similarly named pairs.
      - Code has further instructions and explanaton.
      - Some manual effort is required.
9) MobileUNETR_Framework.py
      - PyTorch Lightning implementation of MobileUNETR model for binary segmentation.
      - Requires mobileunetr_xxs. To do this, access the following github (https://github.com/OSUPCVLab/MobileUNETR). Download the contents of the 'architectures' folder into the same directory.
10) SegFormer_Framework.py
      - Pytorch Lightning implementation of SegFormer model for binary segmentation.
      - Should not require any manual work. It will access the model and pretrained weights from Huggingfgace.
11) Hector J. Cotto Ortiz (2025) Exploring Automated Lunar Crater detection Using Gravity and Optical Data Modalities with Lightweight Segmentation Models.pdf
      - Thesis developed with the results of this work
12) Tnsorboard Results.7z
      - Zipped file containing tensorboard results of every iteration across each Dataset. It contains a folder for each data type, and each further contains a folder for MobileUNETR and SegFormer results. 

NOTE: You are welcome to use different data intended for binary segmentation, but if you do, follow the folder structure established in the repository (Or manually change the paths in the code).
Another important thing of note is that this assumes the Image/label pairs files are all similarly named (e.g. Regional Image Splits contains files with same names, but different data).
Because I kept my three data types (Gravity/Optical, Gravity and Optical) physically separtate, I didn't give each a name, and this allowed me to train with a different dataset by simply changing the main folder math name. 
As such, if using different data you just need to make sure that Image and label pairs are named similarly, wth an appended '_label' to differentiate (e.g.  'example1_tif' and 'example1_label.tif')

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Folder structure used in this work. Each Dataset was differentiated by a number, and inside it contained 5 main folders. 3 for the Training, Validation and Test regions, and 2 to store training infomation from MopbileUNETR and SegFormer.
![Folder Structure Used in this work](https://github.com/user-attachments/assets/974ae385-ce68-49d8-a570-4c1c9e8e6982)


Regional Image and Label showing the Training, Validation and Test areas
![Training regions Optical and Label_Annotated](https://github.com/user-attachments/assets/6e4f0770-54dd-4fbf-a79b-0b82f9fcd419)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Instructions to use: 
1) Create a folder structure as shown for however many datasets you wish to use. In this repository there are 3.
      - For example, you create a folder named Optical Dataset, inside it you will have
              - A Training, Validation and Test folder containing the corresponding portion of the image and label. These are found in Regional Image Splits. Every dataset uses the same label.
              - You will also have a folder for MobileUNETR results, and another for SegFormer results. If you only want to use one model, then you only need one folder.
              - If you want to use your own data, that's okay but it should still be separated into Training, Validation and Test folders.
2) Use the Overlap Image Splitter code
      - This code takes an image and performs the following:
              - Split the image into overlapping chunks. You select the overlap percentage as well as the chunk size.
              - Perform image augmentation. There are three preset augmentations, and they're explained within the code.
              - It will name each image and label with the same name, appending the label with '_label' to easily find each counterpart. 
      - Note: This code is hardcoded to use a Training, Validaton and Test image portion. You can comment out whatever you need.
      - Note 2: If your data is already split into image/label pairs you can SKIP THIS STEP.
3) Open MobileUNETR Framework or SegFormer Framework and alter the data paths to correspond to your data. I've tried to make the data paths into variables so you only need to change one. I am going to work on making it so you can choose your own paths at a later time.
4) Provided there isn't something I forgot to tell you, it should work. Please let me know if you encounter problems. A lot of this eas manually set up for my personal work, but since I made it this far I thought I'd make it availabel to use. I will be updating the codes to work more seamlessly with other users and avoid so much manual labor. Unfortunately, unless Im abe to host the prepared dataset there will still be a need to split them yourself in the meantime. That said, I encourage you to try other datasets you may have as long as they're meant for two classes. 

