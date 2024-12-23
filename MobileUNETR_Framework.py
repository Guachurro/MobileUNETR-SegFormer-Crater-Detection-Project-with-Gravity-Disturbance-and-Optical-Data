# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:41:35 2024

@author: Guach
"""

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
import cv2

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
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

#Hugging Face Transformers
import transformers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

#Hugging Face Datasets
import evaluate
#%

#######################BLOCK 2: DATA PROCESSOR, FINETUNER CLASS DEFINITIONS
class SegmentationDatasetCreator(data.Dataset):
    def __init__(self,folder_path,img_list,label_list,imagesize):
        super(SegmentationDatasetCreator,self).__init__()
        self.img_files=img_list #This is the list of image paths.
        self.mask_files=label_list #To store the correspoinding bitmap for each image path
        self.imagesize=imagesize
    def __getitem__(self, index):      
        # Load the image and label using rasterio. read all bands
        crater_img = rasterio.open(self.img_files[index]).read()
        crater_label = rasterio.open(self.mask_files[index]).read()
        
        # Identify the nodata values (-32767) for Image and eliminate them for the label
        meanvalue=crater_img.mean()
        crater_img[crater_img==-32767] = meanvalue
        crater_label[crater_label == -32767] = 0
        
        #Squeeze the label
        crater_label=crater_label.squeeze()
       
        #Resizing crater IMG and Label
        crater_img = np.transpose(crater_img, (1, 2, 0))
        resized_img = cv2.resize(crater_img, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        resized_img=np.transpose(resized_img,(2,0,1))
        resized_label = cv2.resize(crater_label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)        
        
        
        #Apply the transformation to both the image and the label
        crater_img = torch.from_numpy(resized_img)
        crater_label = torch.from_numpy(resized_label)  
        crater_label=crater_label.unsqueeze(0)
        
        inputs=inputs = {
    'pixel_values': crater_img,  # Removing any extra dimensions, if needed
    'labels': crater_label.long()   # Adding channel dimension if single-channel label is required
}

        return inputs

    def __len__(self):
        return len(self.img_files)


#PyTorch Lightning Class Implementation
class MobileUNETRFinetuner(pl.LightningModule):
    def __init__(self,id2label,train_dataloader=None,val_dataloader=None,test_dataloader=None, metrics_interval=5,modelweights=None,hyperparameters=None):
                 super(MobileUNETRFinetuner, self).__init__()
                 
                 self.save_hyperparameters()
                 self.id2label=id2label
                 self.metrics_interval=metrics_interval
                 self.train_dl=train_dataloader
                 self.val_dl=val_dataloader
                 self.test_dl=test_dataloader
                 self.modelweights=modelweights[0]
                 self.freeze_encoder=modelweights[1]
                 self.hyperparameters=hyperparameters
                 self.randomseed=hyperparameters[9]
                 
                 
                 self.test_images=[]
                 self.test_predictions=[]
                 self.test_labels=[]
                 
                 #Number of Classes in Model
                 self.num_classes=len(id2label.keys())
                 self.label2id = {v:k for k,v in self.id2label.items()}
                 
                 self.imageresize=hyperparameters[0]
                 #Model Loaded from Internal Directory
                 self.model=build_mobileunetr_xxs(num_classes=1,image_size=imageresize)
                 
                 #Which Weights were used. 
                 if modelweights is not None:
                     print(f"{modelweights} being applied")
                     self.model.load_state_dict(torch.load(self.modelweights,map_location=torch.device('cpu')))
                     
                 #Prints whether the model is frozen. 
                 if self.freeze_encoder==True:
                    print("Freezing encoder")
                    for param in self.model.encoder.parameters():
                        param.requires_grad = False

                 #Loss function
                 self.lossfunc=DiceLoss()

                 #Functions to calculate metrics
                 self.train_mean_iou = evaluate.load("mean_iou", use_auth_token=None) #Intersection over union is the metric used to determine how well a [redicted mask matches ground truth data
                 self.val_mean_iou = evaluate.load("mean_iou",use_auth_token=None)
                 self.test_mean_iou = evaluate.load("mean_iou",use_auth_token=None)
                 
    def forward(self, images):
        # Pass images through the model
        outputs = self.model(x=images)
        
        return outputs
        
    
    def training_step(self, batch, batch_nb):
        images, masks= batch['pixel_values'], batch['labels']
        logits=self(images) 
       
        if logits.dim() == 3:
            logits = logits.unsqueeze(1).float()  # Add a channel dimension if missing
        
        if logits.shape != masks.shape:
            upsampled_logits=nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:],
                mode="nearest",
                align_corners=False)
        else:
            upsampled_logits=logits
          
        #Activation Function
        predicted = torch.sigmoid(upsampled_logits)
        
        #Loss Function
        loss=self.lossfunc(predicted,masks)
        
        #Thresholding probabilities to obtain classification
        predicted = (predicted > .5).float()
        
        self.train_mean_iou.add_batch(
            predictions=predicted.squeeze().detach().cpu(),
            references=masks.squeeze().detach().cpu()
            )

       
        #Log the Training loss
        self.log('train_loss',loss,on_step=True,on_epoch=True)
      
        return({'loss': loss})
        
    def on_train_epoch_end(self):
        metrics=self.train_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=np.nan,
            reduce_labels=False,
            )
    
        #Overall Pixel Accuracy, considering all classes
        self.log("Train_Pixelwise_Acc", metrics["overall_accuracy"])  # This is the pixel-wise accuracy       
        #Mean IoU between classes
        self.log('Train_mean_iou', metrics["mean_iou"])
        #Mean Pixel Accuracy amongst all classes
        self.log('Train_mean_accuracy', metrics["mean_accuracy"])
    
        #Calculate Class-Specific IoU
        if "per_category_iou" in metrics:
            for class_idx, iou in enumerate(metrics["per_category_iou"]):
                self.log(f'Train_IoU_class_{class_idx}', iou)
        
    def validation_step(self, batch, batch_nb):
        if not hasattr(self, 'validation_outputs'):
            self.validation_outputs = []
        
        images, masks = batch['pixel_values'], batch['labels']
        logits = self(images)

        #Ensure proper dimensions and upsample if necessary
        if logits.dim() == 3:
            logits = logits.repeat(1,3,1,1).float()  # Add a channel dimension if missing
        
        if logits.shape != masks.shape:
            upsampled_logits = nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode="nearest", 
                align_corners=False
                )
        else:
            upsampled_logits=logits
        
        #Activation function
        predicted = torch.sigmoid(upsampled_logits)

        #Loss Function
        loss=self.lossfunc(predicted,masks)
        predicted = (predicted > .5).float()
 
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().squeeze().cpu(),#.numpy(), 
            references=masks.detach().squeeze().cpu()#.numpy()
        )
      
        # Log the validation loss directly, with both on_step and on_epoch
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        
    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=np.nan,
              reduce_labels=False,
          )
        
        #Overall Pixel Accuracy considering all classes
        self.log("Val_Pixelwise_Acc", metrics["overall_accuracy"])
        #Mean IoU between classes
        self.log('Val_mean_iou', metrics["mean_iou"])
        #Mean Pixel Accuracy amongst all classes
        self.log('Val_mean_accuracy', metrics["mean_accuracy"])
        
        #Calculate Class-Specific IoU
        if "per_category_iou" in metrics:
            for class_idx, iou in enumerate(metrics["per_category_iou"]):
                self.log(f'Val_IoU_class_{class_idx}', iou)
                
    def test_step(self, batch, batch_nb):
        if not hasattr(self, 'test_outputs'):
            self.test_outputs = []
        
        
        images, masks = batch['pixel_values'], batch['labels']
        logits = self(images)
        
        
        #Ensure proper dimensions and upsample if necessary
        if logits.dim() == 3:
            logits = logits.unsqueeze(1).float()
        
        if logits.shape != masks.shape:
            upsampled_logits = nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode="nearest", 
                align_corners=False
                )
        else:
            upsampled_logits=logits
        
        #Threshold
        predicted = torch.sigmoid(upsampled_logits)
        
        #Loss Function
        loss=self.lossfunc(predicted,masks)
        predicted = (predicted > .5).float()
        
        self.test_mean_iou.add_batch(
            predictions=predicted.squeeze().detach().cpu(),#.numpy(), 
            references=masks.squeeze().detach().cpu()#.numpy()
        )
        
        #Accumulating Test Images, Predictions and Labels
        self.test_images.append(images.cpu())
        self.test_predictions.append(predicted.cpu())
        self.test_labels.append(masks.cpu())
        
        # Log the test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        metrics=self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=np.nan,
            reduce_labels=False,
            )
        
        #Overall Pixel Accuracy considering all classes
        self.log("Test_Pixelwise_Acc", metrics["overall_accuracy"]) 
        #Mean IoU between classes
        self.log('Test_mean_iou', metrics["mean_iou"])
        #Mean Pixel Accuracy amongst all classes
        self.log('Test_mean_accuracy', metrics["mean_accuracy"])
        
        #Calculate Class-Specific IoU
        if "per_category_iou" in metrics:
            for class_idx, iou in enumerate(metrics["per_category_iou"]):
                self.log(f'Test_IoU_class_{class_idx}', iou)
                
        #Visualizing Predictions at End Step
        all_images = torch.cat(self.test_images, dim=0)
        all_preds = torch.cat(self.test_predictions, dim=0)
        all_labels = torch.cat(self.test_labels, dim=0)
        
        # Visualize final subset of predictions
        num_images_to_visualize = min(20, all_images.size(0))
        for i in range(num_images_to_visualize):
            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            ax[0].imshow(all_images[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
            ax[0].set_title('Original Image')
            ax[1].imshow(all_labels[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
            ax[1].set_title('True Mask')
            ax[2].imshow(all_preds[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
            ax[2].set_title('Predicted Mask')
            ax[3].imshow(all_preds[i].permute(1, 2, 0).cpu().numpy(), cmap='gray', alpha=0.5)
            ax[3].imshow(all_labels[i].permute(1, 2, 0).cpu().numpy(), cmap='jet', alpha=0.5)
            ax[3].set_title('Overlay')
            plt.show()
        
        
    def configure_optimizers(self):
        print("Entering Optimizers")
        chosenoptimizer=self.hyperparameters[1]
        learningrate=self.hyperparameters[2]
        eps=self.hyperparameters[3]
        weightdecay=self.hyperparameters[4]
        betas=self.hyperparameters[5]
        asmgrads=self.hyperparameters[6]
        print(f"Chosen Optimizer|{self.hyperparameters[1]}")
        
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
            sgdmomentum=self.hyperparameters[7]
            sgdnesterov=self.hyperparameters[8]
            optimizer = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=learningrate, momentum=sgdmomentum, weight_decay=weightdecay,nesterov=sgdnesterov)
            print(f"Hypeparameters are:LR {learningrate} | decay {weightdecay} | Momentum {sgdmomentum} | Nesterov {sgdnesterov} ")    
        
        print("Defining scheduler")
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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Dice Loss for binary segmentation.
        
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Common values are between 1.0 and 1e-6.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Forward pass for Dice Loss.

        Args:
            pred (torch.Tensor): Predicted mask of shape [B, 1, H, W], where B is the batch size.
            target (torch.Tensor): Ground truth mask of shape [B, 1, H, W].
        
        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        if pred.shape[1] == 2:  # Check if there are two channels (background and foreground)
            pred = pred[:, 1, :, :]  # Use only the foreground channel
        
        # Flatten the tensors to compute Dice coefficient over all pixels
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # Dice Loss is 1 - Dice coefficient
        dice_loss = 1 - dice_coeff
        return dice_loss

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
#IMPORTANT: CHOOSE THE MODEL WEIGHTS, PARAMETERS, OPTIMIZERS, WHETHER YOU WISH TO FREEZE ENCODER, ETC BEFORE RUNNIN
HyperparametersChosen=[]
#How many times the experiment is repeated
Attempts=50
tensorversion=-1 #Necessary for Iterations and TensorLogs to end up with the same number
Run=2 #Create a folder for a new set of Iterations
#Choose the Folder for the dataset you want. Since we have similarly named folders, all we need is a number
dataset='Dataset Number'

for k in range(Attempts):
    print(f"Starting loop {k}")
    
    #Random Seed Selection, Image Size and Torch Device. 
    seed=random.randint(0, 2**32 - 1) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    imageresize=128 #Resize Images to this value
    interval=2 #Log every number of steps/batches
    torch.device("cpu")
    
    
    #1) Model Weights 
    weights1="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/isic_2016_pytorch_model.bin"
    weights2="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/isic_2017_pytorch_model.bin"
    weights3="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/isic_2018_pytorch_model.bin"
    weights4="F:/Academics/OSU/Thesis Documents/MobileUNETR-main/weights/final_model_files/ph2_pytorch_model.bin"
    chosenweights=random.choice([weights1,weights2,weights3,weights4])
    
    #2) Randomize Encoder Freezing.
    freezenc=random.choice([True,False])
    
    #Parameter passed to PyTorch Lightning
    modelinfo=[chosenweights,freezenc]
    
    #Hyperparameter Information
    batch_size=random.choice([8,16,24,32,40]) #Random Batch size
    batch_size_eval=64 #Not necessary to change, it just makes it faster
    epochs=100 #Redefined later 
    
    #Optimizer Selection
    Optimizer1='Adam'
    Optimizer2='AdamW'
    Optimizer3='SGD'
    chosenoptimizer=random.choice([Optimizer1,Optimizer2,Optimizer3])
    
    #Optimizer Hyperparameters. These apply to every Optimizer. 
    factor=random.choice([1,2,3,4,5])
    rate=random.choice([1e-2,1e-3,1e-4,1e-5])
    learningrate=rate*factor
    eps=1e-8
    weightfactor=random.choice([1,2,3,4,5])
    decay=random.choice([.1,.01,.001])
    weightdecay=weightfactor*decay#Random weight decay
            
    #Adam Hyperparameters
    adambetas=(.9,.999)
    amsgrad=random.choice([True,False]) #If True it can improve convergence
            
    #SGD Hyperparameters
    momfactor=random.choice([1,2,3,4,5,6,7,8,9])
    momentum=random.choice([.1,.01]) #Adds fraction of previous update, could help if plateaus
    sgdmomentum=momfactor*momentum#Random Momentum
    nesterov=random.choice([True,False]) #If true it operates as amsgrad but instead of looking backwards, it looks forward

    hyperparams=[imageresize,chosenoptimizer,learningrate,eps,weightdecay,adambetas,amsgrad,sgdmomentum,nesterov,seed]

    #Attempt
    tensorversion=k
    HyperparametersChosen.append(hyperparams)
    print(f"HYPERPARAMETERS {hyperparams}")
    
    #Image and Tensorboard Information
    #Path Containing Root folders for each Dataset
    folder_path=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {dataset}/"

    #Where Checkpoint and Tensorboard information will be stored
    CheckpointPath=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {dataset}/ModelCheckpoint_Mobileunetr/Attempt {Run}/Iteration {tensorversion}/"
    CheckpointPathT=f"F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset {dataset}/ModelCheckpoint_Mobileunetr/Attempt {Run}/"
    Tensorname='MobileUNETR'
    
    #Make sure directories exist
    if not os.path.exists(CheckpointPath):
        print(f"Creating {CheckpointPath}")
        os.makedirs(CheckpointPath)

    #Create list of all Image/Label pairs and shuffle them. Assumes a specific dataset structure. 
    #First images are obtained and shuffled
    train_img_files=glob.glob(os.path.join(folder_path,'Training/Image/','*.tif')) 
    random.shuffle(train_img_files)
    
    val_img_files=glob.glob(os.path.join(folder_path,'Validation/Image/','*.tif'))
    random.shuffle(val_img_files)
    
    test_img_files=glob.glob(os.path.join(folder_path,'Test/Image/','*.tif'))  
    random.shuffle(test_img_files)

    #The appropriate labels for each image is obtained. No need to shuffle again. 
    train_mask_files=[]
    for img_path in train_img_files: 
            train_mask_files.append(os.path.join(folder_path+'Training/Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))
            
    val_mask_files=[]
    for img_path in val_img_files: 
            val_mask_files.append(os.path.join(folder_path+'Validation/Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))
            
    test_mask_files=[]
    for img_path in test_img_files: 
            test_mask_files.append(os.path.join(folder_path+'Test/Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))
    
    #Debugging Purposes
    totalsize=len(train_img_files)+len(val_img_files)+len(test_img_files)
    
    #Read and format image/label pairs into tensor datasets
    train_dataset=SegmentationDatasetCreator(folder_path,train_img_files,train_mask_files,imageresize)
    validation_dataset=SegmentationDatasetCreator(folder_path,val_img_files,val_mask_files,imageresize)
    test_dataset=SegmentationDatasetCreator(folder_path,test_img_files,test_mask_files,imageresize)

    #Create DataLoaders to pass to the model
    train_dataloader1=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)#,num_workers=workers)
    val_dataloader1=DataLoader(validation_dataset,batch_size=batch_size)#,num_workers=workers)
    test_dataloader1=DataLoader(test_dataset,batch_size=batch_size)#,num_workers=workers)

    #Dictionary Information. Here's where classes are defined. 
    Label_dictionary= {
        0: "background",
        1: "crater"
        }

    #This defines the PyTorch Lightning Modules to later pass it onto the trainer
    segformer_finetuner = MobileUNETRFinetuner(
        id2label=Label_dictionary, 
        train_dataloader=train_dataloader1, 
        val_dataloader=val_dataloader1, 
        test_dataloader=test_dataloader1, 
        metrics_interval=interval,
        modelweights=modelinfo,
        hyperparameters=hyperparams
        )

    #Stopgap measures
    early_stop_callback = CustomEarlyStopping(
        monitor="val_loss", #Considers improvement based on validation loss
        min_delta=0.00, #Any ammount of improvement, regardless of how small, is considered an improvement. 
        patience=10, #How many consecutive epochs without improvement are needed
        verbose=True, 
        mode="min",
        )
    
    #Defining Checkpoint Saver and TensorBoard
    checkpoint_callback = ModelCheckpoint(dirpath=CheckpointPath,
                                          filename="{epoch:02d}--{val_loss:.2f}",
                                          monitor="val_loss",
                                          save_last=True, #Always saves latest recorded epoch
                                          save_weights_only=False,
                                          save_top_k=5, #Save Top 5 epochs with smallest validation losses
                                          )

    tensorboard_logger= TensorBoardLogger(CheckpointPathT,name=Tensorname,version=tensorversion)

    # Instantiate the progress bar callback
    progress_bar = TQDMProgressBar()

    #Line of code necessary to visualize Tensorbboard locally.
    print(f"Launch TENSORBOARD: tensorboard --logdir={CheckpointPathT}")
    trainer = pl.Trainer(
        accelerator="cpu", 
        callbacks=[early_stop_callback, checkpoint_callback, progress_bar],
        max_epochs=epochs,
        val_check_interval=len(train_dataloader1),
        logger=tensorboard_logger,
        log_every_n_steps=interval,
        )

    #Initialize the Training Session
    trainer.fit(segformer_finetuner)
    
    print("Going to new loop")

#%% Example to visualize results using test dataset
#TESTING MOBILEUNETR DTASET 
seed=1638565519 #Choose value that was in the same run
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
imageresize=128

#Change this with your own folder.
folder_path="F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 6/"

#Obtain Test Images and Shuffle
test_img_files=glob.glob(os.path.join(folder_path,'Test/Image/','*.tif'))  
#Obtain respective labels
random.shuffle(test_img_files)          
test_mask_files=[]
for img_path in test_img_files: 
    test_mask_files.append(os.path.join(folder_path+'Test/Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))


#Read and Format Images
test_dataset=SegmentationDatasetCreator(folder_path,test_img_files,test_mask_files,128)

#Create DataLoaders
test_dataloader1=DataLoader(test_dataset,batch_size=40)

#Define the PyTorch Lightning Framework the selected checkpoint
testmodel=MobileUNETRFinetuner.load_from_checkpoint("F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 6/ModelCheckpoint_Mobileunetr/Attempt 1/Iteration 28/epoch=45--val_loss=0.36.ckpt")

#Define a new Trainer and change the model to eval mode. 
#This freezes encoder and decoder. Meaning the model will only extract information and generae predictions without altering existing weights.  
Test_Trainer=pl.Trainer(accelerator="cpu")
testmodel.eval()

#Initiate Session
Test_Trainer.test(testmodel, dataloaders=test_dataloader1)


#%% Example to visualize results using test dataset
#TESTING MOBILEUNETR DTASET Optical
seed=1638565519 #Choose value that was in the same run
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
imageresize=128

#Change this with your own folder.
folder_path="F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/"

#Obtain Test Images and Shuffle
test_img_files=glob.glob(os.path.join(folder_path,'Test/Image/','*.tif'))  
#Obtain respective labels
random.shuffle(test_img_files)          
test_mask_files=[]
for img_path in test_img_files: 
    test_mask_files.append(os.path.join(folder_path+'Test/Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))


#Read and Format Images
test_dataset=SegmentationDatasetCreator(folder_path,test_img_files,test_mask_files,128)

#Create DataLoaders
test_dataloader1=DataLoader(test_dataset,batch_size=40)

#Define the PyTorch Lightning Framework the selected checkpoint
testmodel=MobileUNETRFinetuner.load_from_checkpoint("F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 7/ModelCheckpoint_Mobileunetr/Attempt 1/Iteration 35/epoch=20--val_loss=0.48.ckpt")

#Define a new Trainer and change the model to eval mode. 
#This freezes encoder and decoder. Meaning the model will only extract information and generae predictions without altering existing weights.  
Test_Trainer=pl.Trainer(accelerator="cpu")
testmodel.eval()

#Initiate Session
Test_Trainer.test(testmodel, dataloaders=test_dataloader1)

#%% Example to visualize results using test dataset
#TESTING MOBILEUNETR DTASET Fusion
seed=1638565519 #Choose value that as in the same run
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
imageresize=128

#Change this with your own folder.
folder_path="F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 5/"

#Obtain Test Images and Shuffle
test_img_files=glob.glob(os.path.join(folder_path,'Test/Image/','*.tif'))  
#Obtain respective labels
random.shuffle(test_img_files)          
test_mask_files=[]
for img_path in test_img_files: 
    test_mask_files.append(os.path.join(folder_path+'Test/Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))


#Read and Format Images
test_dataset=SegmentationDatasetCreator(folder_path,test_img_files,test_mask_files,128)

#Create DataLoaders
test_dataloader1=DataLoader(test_dataset,batch_size=40)

#Define the PyTorch Lightning Framework the selected checkpoint
testmodel=MobileUNETRFinetuner.load_from_checkpoint("F:/Academics/OSU/Thesis Documents/Images and Labels/Dataset 5/ModelCheckpoint_Mobileunetr/Attempt 1/Iteration 34/epoch=27--val_loss=0.40.ckpt")

#Define a new Trainer and change the model to eval mode. 
#This freezes encoder and decoder. Meaning the model will only extract information and generae predictions without altering existing weights.  
Test_Trainer=pl.Trainer(accelerator="cpu")
testmodel.eval()

#Initiate Session
Test_Trainer.test(testmodel, dataloaders=test_dataloader1)

