# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:11:37 2024

@author: Guach
"""

#%%BLOCK 1: MODULES, VARIABLE DEFINITION AND DATA PATHS

#Creating Custom Dataloader for Semantic Segmentation Dataset
#This loader will assume two different folders within one same folder_path directory: Images and Labels, respectively
import sys
import os
import glob
import random

sys.path.append("F:/Academics/OSU/Thesis Documents/MobileUNETR-main/architectures")


#Third Party libraries
import numpy as np
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns

#MobileUNETR
from mobileunetr import build_mobileunetr_s,build_mobileunetr_xs,build_mobileunetr_xxs

import einops

#SciKit-Learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#Pytorch and Torch-based imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

#Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

#Hugging Face Transformers
import transformers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

#Hugging Face Datasets
import evaluate

#LookAhead optimizer add-on
from ranger import Ranger

#IMPORTANT: CHOOSE THE MODEL WEIGHTS, PARAMETERS, OPTIMIZERS, WHETHER YOU WISH TO FREEZE ENCODER, ETC BEFORE RUNNIN

#Model Weights (Mostly choose one to start it then dont change it again)
weights1="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/isic_2016_pytorch_model.bin"
weights2="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_filesqwe/isic_2017_pytorch_model.bin"
weights3="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/isic_2018_pytorch_model.bin"
weights4="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/ph2_pytorch_model.bin"
#Choose weights for the model
chosenweights=weights1

#Make True to freeze encoder(You can change this later if you want)
freezenc=True
#Passed to FineTuner
modelinfo=[chosenweights,freezenc]

#model hyperparameters
#DataPre Processing
seed=42 #Seed should stay for reproducibility
imageresize=128 #I'd avoid changing this
torch.device("cpu")

#Hyperparameter Information
batch_size=8
batch_size_eval=64 #Not necessary to change, it just makes it faster
epochs=5 #Redefined later 

#Optimizer: Also change any relevant information
Optimizer1='Adam'
Optimizer1='AdamW'
Optimizer1='SGD'
chosenoptimizer=Optimizer1

#Optimizer Hyperparameters
learningrate=1e-5
eps=1e-8
weightdecay=.001

#Adam Optimizers
adambetas=(.9,.999)
amsgrad=False #If True it can improve convergence

#SGD
sgdmomentum=0.1 #Adds fraction of previous update, could help if plateaus
nesterov=False #If true it operates as amsgrad but instead of looking backwards, it looks forward


#True to use LookAhead
uselookahead=True

hyperparams=[imageresize,chosenoptimizer,learningrate,eps,weightdecay,adambetas,amsgrad,sgdmomentum,nesterov,uselookahead]


#True to use LookAhead
uselookahead=True
if uselookahead==True:
    k=5 #How many updates are done by fast optimizer before slow weights are applied
    alpha=0.5 #Slow weights factor (Between 0 and 1)
    hyperparams=[imageresize,chosenoptimizer,learningrate,eps,weightdecay,adambetas,amsgrad,sgdmomentum,nesterov,uselookahead,k,alpha]


#Dataset Information
dataset=1
maptype=660
CentX='13E'
CentY='0N'

#Attempt
tensorversion=0

#Image and Tensorboard Information
folder_path=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {dataset}/{maptype}/"
GravityImage=rasterio.open(f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {dataset}/{CentX}_{CentY}_30.0by30.0Area_{maptype}.tif").read(1)
#Where information will be stored
CheckpointPath=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {dataset}/ModelCheckpoint_Mobileunetr/Iteration {tensorversion}/"
Tensorname='MobileUNETR'
if not os.path.exists(CheckpointPath):
    print(f"Creating {CheckpointPath}")
    os.makedirs(CheckpointPath)

#information to normalize images
minima=GravityImage.min()
maxima=GravityImage.max()

#%%No need to run this if you don't want to test it
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Load the image
        if self.transform:
            image = self.transform(image)  # Apply transforms (resize, to tensor, etc.)
        return image


modeltest=model=build_mobileunetr_xxs(num_classes=1,image_size=128)
modeltest.load_state_dict(torch.load(chosenweights,map_location=torch.device('cpu')))

# Step 2: Freeze the entire model (both encoder and decoder)
for param in modeltest.parameters():
    param.requires_grad = False
    
# Step 3: Set the model to evaluation mode
modeltest.eval()

#Load images
imagestest=glob.glob(os.path.join(folder_path,'Image/','*.tif')) 
labeltest=[]
for img_path in imagestest: 
            labeltest.append(os.path.join(folder_path+'Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))


# Step 1: Randomize the data
random.seed(seed)
# Zip the image and mask files together to keep them in sync while shuffling
combinedtest = list(zip(imagestest, labeltest))
random.shuffle(combinedtest)  # Randomize the combined data

# Unzip the shuffled list to get randomized image and mask lists
imagestest, labeltest = zip(*combinedtest)

# Convert back to list (since zip returns tuples)
imagestest = list(imagestest)
labeltest = list(labeltest)
#Grab the first 20% of each set
imagestest=imagestest[:round(len(imagestest)*.01)]
labeltest=labeltest[:round(len(labeltest)*.01)]

#Defined transformation
transformation=torchvision.transforms.Compose([torchvision.transforms.Resize((128,128)),
                                               torchvision.transforms.ToTensor()])

#Initialize a DataLoader
datasettest=InferenceDataset(imagestest,transform=transformation)
dataloadertest=DataLoader(datasettest,batch_size=1,shuffle=False)

print(len(dataloadertest))
for batch in dataloadertest:
    images=batch
    numpyimage=images.squeeze(0)
    numpyimage=numpyimage.permute(1, 2, 0).cpu().numpy()
    plt.imshow(numpyimage, cmap='gray')  # Display the first image in the batch
    plt.title("Image before Inference")
    plt.show()
    outputs=modeltest(images)
    predictions=outputs.argmax(dim=1)
    
    # Assuming `predictions` is the output mask from the model
    plt.imshow(predictions[0].cpu().numpy(), cmap='gray')  # Display the first image in the batch
    plt.show()
    


#%% BLOCK 2: DATA PROCESSOR, FINETUNER CLASS DEFINITIONS
#%Technically this isnt a data loader, but rather a feature extractor? 
class SegmentationDatasetCreator(data.Dataset):
    def __init__(self,folder_path,img_list,feature_extractor,gravityminimum,gravitymaximum):
        super(SegmentationDatasetCreator,self).__init__()
        self.img_files=img_list #This is the list of image paths.
        self.mask_files=[] #To store the correspoinding bitmap for each image path
        for img_path in self.img_files: 
            self.mask_files.append(os.path.join(folder_path,'Label/',os.path.basename(img_path)[:-4]+'_Label'+'.tif')) #ADD LAVBEL AGAIN
        self.feature_extractor=feature_extractor
        self.gravityminimum=gravityminimum
        self.gravitymaximum=gravitymaximum
    def __getitem__(self,index):      
        #Prepares images to pass onto pipeline. We only pass training or validation data at a time. 
        #It will open the image and bitmap(label) for each image specified in img_list.
        #You also have to rescale it by subtracting the minimum value (To bring up to 0) and dividing by (Max Value-Min Value)
        crater_img=rasterio.open(self.img_files[index]).read(1)
        crater_img[crater_img==-32767]=0
        crater_img=(crater_img-self.gravityminimum)/(self.gravitymaximum-self.gravityminimum)
        crater_img=transform(crater_img).repeat(3,1,1)
        #print(f"Crater Img: {crater_img.shape}")
        crater_label=transform(rasterio.open(self.mask_files[index]).read(1))
        crater_label[crater_label==-32767]=0
        
        #Image Processor
        inputs = self.feature_extractor(images=crater_img, segmentation_maps=crater_label, return_tensors="pt")
        
        for k,v in inputs.items():
          inputs[k].squeeze_()
        
        return inputs
    def __len__(self):
        return len(self.img_files)


#% Defining and fine tuning the model 
#This is the actual model, as far as I can understand. It has all the instructions on how to handle the data, training, validation, etc. 
class MobileUNETRFinetuner(pl.LightningModule):
    def __init__(self,id2label,train_dataloader=None,val_dataloader=None,test_dataloader=None, metrics_interval=100,modelweights=None,hyperparameters=None):
                 super(MobileUNETRFinetuner, self).__init__()
                 
                 self.save_hyperparameters()
                 self.id2label=id2label #What identifies which data is meant for training
                 self.metrics_interval=metrics_interval
                 self.train_dl=train_dataloader
                 self.val_dl=val_dataloader
                 self.test_dl=test_dataloader
                 self.modelweights=modelweights[0]
                 self.freeze_encoder=modelweights[1]
                 self.hyperparameters=hyperparameters
                 
                 self.num_classes=len(id2label.keys())#Will take the training Dataset and identify how many different pixels there are? 
                 self.label2id = {v:k for k,v in self.id2label.items()}
                 
                 #Hyperparam[0]=imageresize
                 self.model=build_mobileunetr_xxs(num_classes=1,image_size=hyperparameters[0])
                 if modelweights is not None:
                     print(f"{modelweights} being applied")
                     modeltest.load_state_dict(torch.load(self.modelweights,map_location=torch.device('cpu')))
                     
                 if self.freeze_encoder==True:
                    # Step 2: Freeze the entire model (both encoder and decoder)
                    for param in modeltest.encoder.parameters():
                        param.requires_grad = False

                 #Loss function
                 self.lossfunc=DiceLoss()

                
                 self.train_mean_iou = evaluate.load("mean_iou", use_auth_token=None) #Intersection over union is the metric used to determine how well a [redicted mask matches ground truth data
                 self.val_mean_iou = evaluate.load("mean_iou",use_auth_token=None)
                 self.test_mean_iou = evaluate.load("mean_iou",use_auth_token=None)
                 
    def forward(self, images, masks=None):
        # Pass images through the model
        outputs = self.model(x=images)
        #print(f"Logits requires_grad: {outputs.requires_grad}")
        # If masks are provided (during training), return both outputs and masks for loss computation
        if masks is not None:
            return outputs, masks
    
        #print('Pass Forward')    
        return outputs
    
    def training_step(self, batch, batch_nb):
            
        images, masks= batch['pixel_values'], batch['labels']
        outputs=self(images, masks) #Is this step calling the forward function? Yes
        loss, logits=outputs[0], outputs[1]
       #print(f"logits requires_grad1: {logits.requires_grad}")
        # Ensure logits are in floating point format
        logits = logits.float()  # Convert logits to float if they aren't already
        logits.requires_grad_(True)

        
        loss=loss.mean()
        if logits.dim() == 3:
            logits = logits.unsqueeze(1).float()  # Add a channel dimension if missing
            
       # print(f"logits requires_grad2: {logits.requires_grad}")

        
        unsampled_logits=nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False)
        #print(f"Unsampled logits requires_grad: {unsampled_logits.requires_grad}")


        
        #NEw Loss Function
        loss=self.lossfunc(unsampled_logits,masks)
        
        predicted=unsampled_logits.argmax(dim=1)
        
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
            )
        
        #Log the Training loss
        self.log('train_loss',loss,on_step=True,on_epoch=True)
        
        if batch_nb % self.metrics_interval == 0:
            
            metrics=self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
                )
            loss=loss.mean() #Because this model doesn't convert loss tensors into single scalars on its own 
            
            self.log('train_mean_iou', metrics["mean_iou"])
            self.log('train_mean_accuracy', metrics["mean_accuracy"])
            
        return({'loss': loss})
        
        
    def validation_step(self, batch, batch_nb):
        #print('Enter Validation step')
        if not hasattr(self, 'validation_outputs'):
            self.validation_outputs = []
        
        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]

        if logits.dim() == 3:
            logits = logits.unsqueeze(1).float()  # Add a channel dimension if missing
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        #NEw Loss Function
        loss=self.lossfunc(upsampled_logits,masks)
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        # Log the validation loss directly, with both on_step and on_epoch
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    
    def on_validation_epoch_end(self):
        #print('Enter calidation epoch end')
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        
        #Store mean iou and accuracy for this epoch.         
        self.log('Val_mean_iou', metrics["mean_iou"])
        self.log('Val_mean_accuracy', metrics["mean_accuracy"])


    def test_step(self, batch, batch_nb):
        if not hasattr(self, 'test_outputs'):
            self.test_outputs = []
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        #NEw Loss Function
        loss=self.lossfunc(upsampled_logits,masks)
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )

        # Log the test loss and return it
        self.log('test_loss', loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        self.log('Test_mean_iou',metrics["mean_iou"])
        self.log('test_mean_accuracy',metrics["mean_accuracy"])
        
    def configure_optimizers(self):
        print("Entering Optimizers")
        learningrate=self.hyperparameters[3]
        eps=self.hyperparameters[4]
        weightdecay=self.hyperparameters[5]
        betas=self.hyperparameters[6]
        asmgrads=self.hyperparameters[7]
        chosenoptimizer=self.hyperparameters[2]
        
        if chosenoptimizer=='Adam':
            print("Using Adam Optimizer")
            optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=learningrate, betas=betas, eps=eps, weight_decay=weightdecay)
            print(f"Hypeparameters are:LR {learningrate} | eps {eps} | decay {weightdecay} | Betas {betas}")
        elif chosenoptimizer=='AdamW':
            print("Using AdamW Optimizer")
            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad], lr=learningrate, betas=betas, eps=eps, weight_decay=weightdecay, amsgrad=asmgrads)
            print(f"Hypeparameters are:LR {learningrate} | eps {eps} | decay {weightdecay} | Betas {betas} | ASMGRAD {asmgrads}")
        elif chosenoptimizer=='SGD':
            sgdmomentum=self.hyperparameters[8]
            sgdnesterov=self.hyperparameters[9]
            optimizer = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=learningrate, momentum=sgdmomentum, weight_decay=weightdecay,nesterov=sgdnesterov)
            print(f"Hypeparameters are:LR {learningrate} | decay {weightdecay} | Momentum {sgdmomentum} | Nesterov {sgdnesterov} ")
            
        UseLookAhead=self.hyperparameters[9]
        if UseLookAhead==True:
            lookaheadK=self.hyperparameters[10]
            lookaheadalpha=self.hyperparameters[11]
            print(f"Adding LookAhead with variables K: {lookaheadK},Alpha: {lookaheadalpha}")
            optimizer=Ranger(optimizer, k=lookaheadK,alpha=lookaheadalpha)
            
            
        # Define the ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # 'min' because we want to reduce LR when val_loss stops decreasing
        factor=0.5,           # Factor by which to reduce the LR (e.g., 0.5 reduces it by half)
        patience=6,           # Number of epochs to wait before reducing the LR
        verbose=True,         # Prints a message when the LR is reduced
        threshold=0.0001,     # Threshold for measuring improvement
        cooldown=0,           # Number of epochs to wait before resuming normal operation after LR is reduced
        min_lr=1e-07          # Minimum LR after reduction
        )
        
        # Return optimizer and the scheduler
        return {
        'optimizer': optimizer, 
        'lr_scheduler': {
            'scheduler': scheduler, 
            'monitor': 'val_loss',  # Reduce LR based on validation loss
            'interval': 'epoch',    # Check every epoch
            'frequency': 1          # Apply every epoch
            }
        }
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl
    

def one_hot_encode(targets, num_classes):
    shape = targets.shape
    print(f" Targets before encode: {shape}")
    one_hot = torch.zeros((shape[0], num_classes, shape[1], shape[2]), device=targets.device)
    return one_hot.scatter_(1, targets.unsqueeze(1), 1)

#This definition does not separate classes
class DiceLoss(nn.Module):  # This works the same as torch.nn.Module
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid for binary segmentation (use softmax for multi-class)
        probs = torch.sigmoid(logits)
        #print(f"Targets: {targets.shape}")
        #rint(f"Logits: {logits.shape}")

            # Binary segmentation: no need for one-hot encoding
        if logits.shape[1] == 1:
            targets = targets.unsqueeze(1)  # Make sure targets have the same shape as logits
        else:
            num_classes = logits.shape[1]
            targets_one_hot = one_hot_encode(targets, num_classes)  # For multi-class, apply one-hot encoding
            targets = targets_one_hot

            
        # Flatten the tensors to calculate Dice on all pixels
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # Calculate intersection
        intersection = (probs * targets).sum()

        # Dice coefficient
        dice_score = (2. * intersection + self.smooth) / torch.clamp((probs.sum() + targets.sum() + self.smooth),min=1e-5)

        # Return Dice loss
        return 1 - dice_score


#Define a class to visualize predictions vs true classes
def visualize_predictions(model, dataloader, num_images=5):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad(): 
        for idx, batch in enumerate(dataloader):
            if idx >= num_images:
                break
            
            images, masks = batch['pixel_values'], batch['labels']

            outputs=model(images)

            print(outputs)

            if isinstance(outputs,tuple):
                logits=outputs[0]
            else:
                logits=outputs

            # Interpolate logits to match the size of masks
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            
            predictions = upsampled_logits.argmax(dim=1)

            # Visualize one of the images and its prediction
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())  # Original Image
            ax[0].set_title('Original Image')
            ax[1].imshow(masks[0].cpu().numpy(), cmap='gray')  # True Mask
            ax[1].set_title('True Mask')
            ax[2].imshow(predictions[0].cpu().numpy(), cmap='gray')  # Predicted Mask
            ax[2].set_title('Predicted Mask')
            plt.show()

#Defining Precision, Recall and F1 Score
def precision_recall_f1(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images, masks = batch['pixel_values'], batch['labels']
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            preds = upsampled_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return precision, recall, f1

#EARLY STOPPING DEFINITION 
class CustomEarlyStopping(EarlyStopping):
    def __init__(self, monitor="val_loss", min_delta=0.00, patience=3, verbose=True, mode="min"):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)

        # Log the monitored metric and the current patience state
        print(f"[EarlyStopping] Current {self.monitor}: {current}")
        print(f"[EarlyStopping] Best {self.monitor} so far: {self.best_score}")
        print(f"[EarlyStopping] Epoch without improvement: {self.wait_count}/{self.patience}")

        # Call the original method to maintain its functionality
        super().on_validation_end(trainer, pl_module)
#%%
#LOAD AND SHUFFLE IMAGES
img_files=glob.glob(os.path.join(folder_path,'Image/','*.tif')) 
mask_files=[]
for img_path in img_files: 
            mask_files.append(os.path.join(folder_path+'Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))

totalsize=len(img_files)

# Step 1: Randomize the data
random.seed(seed)
# Zip the image and mask files together to keep them in sync while shuffling
combined = list(zip(img_files, mask_files))
random.shuffle(combined)  # Randomize the combined data

# Unzip the shuffled list to get randomized image and mask lists
img_files, mask_files = zip(*combined)

# Convert back to list (since zip returns tuples)
img_files = list(img_files)
mask_files = list(mask_files)

# Step 2: Remove test data (first 10% after shuffling)
size = len(img_files)
test_img = img_files[:round(size * 0.1)]
test_mask = mask_files[:round(size * 0.1)]

# Step 3: Split remaining data into training and validation
# Remaining data after removing test set
img_files = img_files[round(size * 0.1):]
mask_files = mask_files[round(size * 0.1):]

# 80% for training, 20% for validation
left = round(len(img_files) * 0.8)

# Train data (first 80% of what's left)
train_img = img_files[:left]
train_mask = mask_files[:left]

# Validation data (remaining 20%)
val_img = img_files[left:]
val_mask = mask_files[left:]

assert (len(test_img)+len(train_img)+len(val_img)==totalsize)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.RandomRotation(degrees=30),
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2)
                                            ])

#% Select feature extractor and instantiate chosen class
extractor=SegformerImageProcessor(do_resize=True, size=imageresize, do_rescale=False, do_normalize=False,do_reduce_labels=False)

#Next we pass the data through the image extractor. 
train_dataset=SegmentationDatasetCreator(folder_path,train_img,extractor,minima,maxima)
validation_dataset=SegmentationDatasetCreator(folder_path,val_img,extractor,minima,maxima)
test_dataset=SegmentationDatasetCreator(folder_path,test_img,extractor,minima,maxima)

#Create DataLoaders
train_dataloader1=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)#,num_workers=workers)
val_dataloader1=DataLoader(validation_dataset,batch_size=batch_size)#,num_workers=workers)
test_dataloader1=DataLoader(test_dataset,batch_size=batch_size)#,num_workers=workers)

#id2label dictionary
Label_dictionary= {
    0: "background",
    1: "crater"
}

#This instantiates the finetuner to later pass it onto the trainer
segformer_finetuner = MobileUNETRFinetuner(
    id2label=Label_dictionary, 
    train_dataloader=train_dataloader1, 
    val_dataloader=val_dataloader1, 
    test_dataloader=test_dataloader1, 
    metrics_interval=50,
    modelweights=modelinfo
)

#Stopgap measures
early_stop_callback = CustomEarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=10, 
    verbose=True, 
    mode="min",
)
#%%
checkpoint_callback = ModelCheckpoint(dirpath=CheckpointPath,
                                      filename="{epoch:02d}--{val_loss:.2f}",
                                      monitor="val_loss",
                                      save_last=True,
                                      save_weights_only=False
                                      )

tensorboard_logger= TensorBoardLogger(CheckpointPath,name=Tensorname,version=tensorversion)

# Instantiate the progress bar callback
progress_bar = TQDMProgressBar()


print(f"Launch TENSORBOARD: tensorboard --logdir={CheckpointPath}")
trainer = pl.Trainer(
    accelerator="cpu", 
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=epochs,
    val_check_interval=len(train_dataloader1),
    logger=tensorboard_logger,
    log_every_n_steps=50,
    resume_from_checkpoint=None
)
trainer.fit(segformer_finetuner)
#%%Fine tuning block to change hyperparameters mid-training
#2)Despite saving all information regarding optimizers and schedulers, we only load the weights and setup an epoch
#2) This assumes youre tracking what epoch your code is in. 
checkpoint = torch.load("checkpoint_with_epoch.pth")
segformer_finetuner.load_state_dict(checkpoint['state_dict'])
start_epoch = 9  # Manually track and set this


#1) You can directly call the hyperparameter variable and change their values. Then you must reinitialize optimizers.
#1) This method avoid backtracking in your code and handles it directly from this block. Everthing else stays the same
#2)The alternative would be to redefine them up on the first block, run that code.
#2)Thenyou have to define segformer_finetuner again and reload the weights.
#2)If using alternate method then move this line to the top of THIS block. 
#3) LEAVE THEM AS IS TO USE DEFAULT CONFIGURATIONS, OTHERWISE CHANGE THEM AS DESIRED
segformer_finetuner.hyperparameters[2] = learningrate  
segformer_finetuner.hyperparameters[4] = weightdecay
segformer_finetuner.hyperparameters[5] = adambetas
segformer_finetuner.hyperparameters[6] = amsgrad
segformer_finetuner.hyperparameters[7] = sgdmomentum
segformer_finetuner.hyperparameters[8] = nesterov
segformer_finetuner.hyperparameters[9] = uselookahead #True or False
segformer_finetuner.train_dl.batch_size=batch_size
if segformer_finetuner.hyperparameters[9]==True:
    segformer_finetuner.hyperparameters[10]=5 #This is k
    segformer_finetuner.hyperparameters[11]=0.5 #Alpha

#Calls current optimizer without initializing. Good for inspection before and after
segformer_finetuner.optimizers() 
#Reinitializes the optimizers by calling it, and stores it. You don't need to store it
optimizer_config = segformer_finetuner.configure_optimizers() 
#Outputsthe curren toptimizer without reinitializing
segformer_finetuner.optimizers() 

#1)You changed the weights,manually put the epoch hyperparameters, reinitialized optimizers and schedulers
#1)Now you're calling the object youve modified again with the changed variables. 
#1)The advantage is that your checkpoints contain actual information on the optimizers and schedulers, we just dont load 
#1)It back into the model
trainer.fit(segformer_finetuner)

#%% Visual inspection
visualize_predictions(segformer_finetuner, segformer_finetuner.test_dataloader(), num_images=len(test_img))
