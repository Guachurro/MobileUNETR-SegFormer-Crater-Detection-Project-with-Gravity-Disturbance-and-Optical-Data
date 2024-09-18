"""
Created on Sun Sep  8 18:37:42 2024

@author: Guach
"""
#%%Custom Data Loader attempt number 2
#Creating Custom Dataloader for Semantic Segmentation Dataset
#This loader will assume two different folders within one same folder_path directory: Images and Labels, respectively
import os
import glob
import random

#Third Party libraries
import numpy as np
from PIL import Image

#Pytorch and Torch-based imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torch.utils.data import random_split

#Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

#Hugging Face Transformers
import transformers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

#Hugging Face Datasets
import evaluate

totalsize=1124
torch.device("cpu")
folder_path="F:/Academics/OSU/Thesis Documents/Images and Labels/"


#Class defined to help interpolate images where there is nodata value stored. 
def interpolate_pixel(data, mask):
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            if mask[i, j]:
                neighbors = data[i-1:i+2, j-1:j+2]
                data[i, j] = np.mean(neighbors[neighbors != -32676])
        return data


#%
#%Technically this isnt a data loader, but rather a feature extractor? 
class SegmentationDatasetCreator(data.Dataset):
    def __init__(self,folder_path,img_list,feature_extractor):
        super(SegmentationDatasetCreator,self).__init__()
        self.img_files=img_list #This is the list of image paths.
        self.mask_files=[] #To store the correspoinding bitmap for each image path
        for img_path in self.img_files: 
            self.mask_files.append(os.path.join(folder_path,'Label/',os.path.basename(img_path),'_Label'))
        
    def __getitem__(self,index):      
        #Prepares images to pass onto pipeline. We only pass training or validation data at a time. 
        #It will open the image and bitmap(label) for each image specified in img_list.
        data=Image.open(self.img_files[index])
        label=Image.open(self.mask_files[index])
        
        inputs = self.feature_extractor(data, label, return_tensors="pt")
        #A feature extractor can replace the next line
        #return torch.from_numpy(data).float(), torch.from_numpy(label).float() #Turned into tensor tuple to pass on to the pipeline? 
        for k,v in inputs.items():
          inputs[k].squeeze_()
        
        return inputs
    def __len__(self):
        return len(self.img_files)
    
    
    
#% Defining and fine tuning the model 
#This is the actual model, as far as I can understand. It has all the instructions on how to handle the data, training, validation, etc. 
class SegformerFinetuner(pl.LightningModule):
    def __init__(self,id2label,train_dataloader=None,val_dataloader=None,test_dataloader=None, metrics_interval=100):
                 super(SegformerFinetuner, self).__init__()
                 self.id2label=id2label #What identifies which data is meant for training
                 self.metrics_interval=metrics_interval
                 self.train_dl=train_dataloader
                 self.val_dl=val_dataloader
                 self.test_dl=test_dataloader
                 
                 self.num_classes=len(id2label.keys())#Will take the training Dataset and identify how many different pixels there are? 
                 self.label2id = {v:k for k,v in self.id2label.items()}
                 
                 
                 self.model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", #This determined which weights to bring to the model
                                                                             return_dict=False,
                                                                             num_labels=self.num_classes, #How many classes the mdoel will segment
                                                                             id2label=self.id2label, #Matches pixels to mask
                                                                             label2id=self.label2id,#Matches mask values to pixels
                                                                             ignore_mismatched_sizes=True,
                                                                             )
                 self.train_mean_iou = evaluate.load("mean_iou") #Intersection over union is the metric used to determine how well a [redicted mask matches ground truth data
                 self.val_mean_iou = evaluate.load("mean_iou")
                 self.test_mean_iou = evaluate.load("mean_iou")
                 
    def forward(self, images, masks):
        outputs=self.model(pixel_values=images, labels=masks) #Passes Images and masks to model defined in Init
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        images, masks= batch['pixel_values'], batch['labels']
        outputs=self(images, masks)
        loss, logits=outputs[0], outputs[1]
        
        unsampled_logits=nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False)
        
        predicted=unsampled_logits.argmax(dim=1)
        
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpu()
            )
        
        if batch_nb % self.metrics_intervals == 0:
            
            metrics=self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
                )

            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
            for k,v in metrics.items():
                self.log(k,v)
                
            return(metrics)
        else:
            return({'loss': loss})
        
    def validation_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        return({'val_loss': loss})
    
    def validation_epoch_end(self, outputs):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)

        return metrics
    
    def test_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
            
        return({'test_loss': loss})
    
    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl



#The steps that need to be taken to interpolate the images. 
# Load the image
#image = Image.open('your_image.png')
#image_data = np.array(image)

# Identify pixels with the value -32676
#mask = (image_data == -32676)

#Class defined to help interpolate images where there is nodata value stored. 
def interpolate_pixel(data, mask):
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            if mask[i, j]:
                neighbors = data[i-1:i+2, j-1:j+2]
                data[i, j] = np.mean(neighbors[neighbors != -32676])
    return data













#%%
#This block is the first actualpiece of code, it will take the images in my folder_path and seperate them into Training, validation and testing.
#Each run it generates a diffrent set of validation/training but keeps the exact same testing group. 
img_files=glob.glob(os.path.join(folder_path,'Image','*.tif')) 
mask_files=[]
for img_path in img_files: 
            mask_files.append(os.path.join(folder_path+'Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))

#Split the image paths into Train, Validation and test.
#Take out test data first(Last .10% of dataset)
size=len(img_files)
test_img=img_files[size-round(size*.1):] 
test_mask=mask_files[size-round(size*.1):]

#Remaining images. The masks are acquired within the data loader
img_files=img_files[:size-round(size*.1)]

#Shuffle current list and determing training/validation size
size=len(img_files)
random.shuffle(img_files)

train_size=round(size*.8)

#Pass one of these two to the dataloader to pass on to pipeline.
train_img=img_files[:train_size]
val_img=img_files[train_size:]

assert (len(test_img)+len(train_img)+len(val_img)==totalsize)



#%% Select feature extractor and instantiate chosen class
#Instantiate the predefined class with the feature extractor
extractor=SegformerImageProcessor(do_resize=True)#Reduce labels will change any pixel above the defined classes to a default class, such as 0. #extractor.do_reduce_labels='False'
#extractor.size=512 #Size to resize images and their labels

#Run each image batch through the Dataset Creator
#Passing the images through the feature extractor oto be put into the correct format is what creates the 'Dataset' 
train_dataset=SegmentationDatasetCreator(folder_path,train_img,extractor)
validation_dataset=SegmentationDatasetCreator(folder_path,val_img,extractor)
test_dataset=SegmentationDatasetCreator(folder_path,test_img,extractor)

#Number of images at a time
batch_size=10
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader=DataLoader(validation_dataset,batch_size=batch_size)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size)




#% Instantiate the fine tuned model
#in this case instantiate means to define it. 
segformer_finetuner = SegformerFinetuner(
    id2label=train_dataset, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    test_dataloader=test_dataloader, 
    metrics_interval=10,
)





#%% Adding stop based on validation loss to prevent overfitting. Create Pytorch lightning trainer and start training????
#Actual training step? 
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=3, 
    verbose=False, 
    mode="min",
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

trainer = pl.Trainer(
    gpus=1, 
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=500,
    val_check_interval=len(train_dataloader),
)

trainer.fit(segformer_finetuner)


#Questions. Wehere exactly do I give it the data that it needs?
#Do I need to install the actual model, or will this code itself work? 
