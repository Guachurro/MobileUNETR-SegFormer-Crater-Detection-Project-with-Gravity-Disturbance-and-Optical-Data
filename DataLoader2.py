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
#import multiprocessing

#Third Party libraries
import numpy as np
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns

#SciKit-Learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#Pytorch and Torch-based imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision

#Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

#Hugging Face Transformers
import transformers
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

#Hugging Face Datasets
import evaluate

#Here we select things like batch size, worker numbers and how many images total. 
batch_size=2
batch_size_eval=1
Image_pass=100
workers=4
torch.device("cpu")

folder_path="F:/Academics/OSU/Thesis Documents/Images and Labels/"
GravityImage=rasterio.open("F:/Academics/OSU/Thesis Documents/Images and Labels/0N_13E_GravtityMap.tif").read(1)
minima=GravityImage.min()
maxima=GravityImage.max()
print(GravityImage.min())
print(GravityImage.max())


#Note: For now, NoData values in the image were turned to 0. 
#%
#%Technically this isnt a data loader, but rather a feature extractor? 
class SegmentationDatasetCreator(data.Dataset):
    def __init__(self,folder_path,img_list,feature_extractor):
        super(SegmentationDatasetCreator,self).__init__()
        self.img_files=img_list #This is the list of image paths.
        self.mask_files=[] #To store the correspoinding bitmap for each image path
        for img_path in self.img_files: 
            self.mask_files.append(os.path.join(folder_path,'Label/',os.path.basename(img_path)[:-4]+'_Label.tif'))
        self.feature_extractor=feature_extractor
    def __getitem__(self,index):      
        #Prepares images to pass onto pipeline. We only pass training or validation data at a time. 
        #It will open the image and bitmap(label) for each image specified in img_list.
        #You also have to rescale it by subtracting the minimum value (To bring up to 0) and dividing by (Max Value-Min Value)
        crater_img=((rasterio.open(self.img_files[index]).read(1))+312.35025)/(815.9169+312.35025)
        crater_img=transform(crater_img).repeat(3,1,1)
        crater_label=transform(rasterio.open(self.mask_files[index]).read(1))


        #Image Processor
        inputs = self.feature_extractor(images=crater_img, segmentation_maps=crater_label, return_tensors="pt")

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
                                                                             use_auth_token=None
                                                                             )
                 self.train_mean_iou = evaluate.load("mean_iou", use_auth_token=None) #Intersection over union is the metric used to determine how well a [redicted mask matches ground truth data
                 self.val_mean_iou = evaluate.load("mean_iou",use_auth_token=None)
                 self.test_mean_iou = evaluate.load("mean_iou",use_auth_token=None)

                 #self.validation_outputs=[]
                 #self.test_outputs=[]
                 
    def forward(self, images, masks=None):
        if masks is not None:
            outputs = self.model(pixel_values=images, labels=masks)
        else:
            outputs=self.model(pixel_values=images, labels=masks) #Passes Images and masks to model defined in Init
        
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        images, masks= batch['pixel_values'], batch['labels']
        outputs=self(images, masks) #Is this step calling the forward function? 
        loss, logits=outputs[0], outputs[1]
        
        unsampled_logits=nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False)
        
        predicted=unsampled_logits.argmax(dim=1)
        
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
            )
        
        if batch_nb % self.metrics_interval == 0:
            
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
        if not hasattr(self, 'validation_outputs'):
            self.validation_outputs = []
        
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
        #Store outputs
        self.validation_outputs.append({'val_loss':loss})

        #No need to return the output because I'm storing them in a variable. 
        #return({'val_loss': loss})
    
    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        
        avg_val_loss = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]

        #Log the metrics
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)
        #Clear them after storing
        self.validation_outputs.clear()
        #return metrics
    
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
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )

        self.test_outputs.append({'test_loss': loss})

        #No need to return it since Im saving it elsewhere
        #return({'test_loss': loss})
    
    def on_test_epoch_end(self):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        #log the metrics
        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)

        self.test_outputs.clear()
        #return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl


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

#Define the confusion Matrix to visualize distribution of predicted classes vs True Classes

def plot_confusion_matrix(model, dataloader):
    model.eval() #Change model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images, masks = batch['pixel_values'], batch['labels'] #It will give the distributions by considering batches as one pile? 
            outputs = model(images) #Grab the images from the Model
            if isinstance(outputs, tuple): #Make sure they're not tuples so we can grab the logits
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

    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(model.num_classes)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show() #This will be what the function returns

#Define Mean Intersection over Union for a single class
def mean_iou_single_class(model, dataloader):
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

            #Get prediction as class with max score
            preds = upsampled_logits.argmax(dim=1).cpu().numpy()

            #Flatten for IoU calculation
            all_preds.extend(preds.flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

    #Convert to numpy array
    all_preds=np.array(all_preds)
    all_labels=np.array(all_labels)

    # Debug: Check unique values in predictions and labels
    print(f"Unique values in predictions: {np.unique(all_preds)}")
    print(f"Unique values in labels: {np.unique(all_labels)}")

    
    #Convert predictions and labels to boolean
    #Assumes that Class 1 is that of interest. This works because I only have a single class. 
    #It's taking every single prediction and label and checking if their value is 1, or not. Their intersections are predictions that were correct. 
    preds_bool=all_preds==1
    labels_bool=all_labels==1

    #Intersection and union
    intersection = ((all_preds == 1) & (all_labels == 1)).sum()
    union = ((all_preds == 1) | (all_labels == 1)).sum()

    # Debug: Check values of intersection and union
    print(f"Intersection: {intersection}")
    print(f"Union: {union}")

    
    #Avoid 0 division
    if union==0:
        iou=float('nan')
    else:
        iou = intersection/union
    
    return iou

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
#%%
#This block is the first actualpiece of code, it will take the images in my folder_path and seperate them into Training, validation and testing.
#Each run it generates a diffrent set of validation/training but keeps the exact same testing group. 
img_files=glob.glob(os.path.join(folder_path,'Image/','*.tif')) 
mask_files=[]
for img_path in img_files: 
            mask_files.append(os.path.join(folder_path+'Label/'+os.path.basename(img_path)[:-4]+'_Label.tif'))
img_files=img_files[0:Image_pass]
mask_files=mask_files[0:Image_pass]
totalsize=len(img_files)
#print('All Images'), print(img_files, totalsize)


#Remove test data. Taken from the first 10%
size=len(img_files)
test_img=img_files[:round(size*.1)] 
test_mask=mask_files[:round(size*.1)]
#print('Test Image'), print(test_img)
#print('Test Mask'), print(test_mask)

#Count how many are left, divide 80% Train and 20% Validation
img_files=img_files[round(size*.1):]
left=round(len(img_files)*.8)
#Train data taken from first 80% of what's left
train_img=img_files[:left]
#Validatio taken from remaining 20%
val_img=img_files[left:]
#print('Train Images'), print(train_img, len(train_img))
#print('Validation Images'), print(val_img, len(val_img))

assert (len(test_img)+len(train_img)+len(val_img)==totalsize)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#% Select feature extractor and instantiate chosen class
#Initiatiling Image Processor with required arguments. Rescaling will be done manually by dividing images by 255
extractor=SegformerImageProcessor(do_resize=True, size=512, do_rescale=False, do_normalize=False,do_reduce_labels=False)

#Next we pass the data through the image extractor. 
train_dataset=SegmentationDatasetCreator(folder_path,train_img,extractor)
validation_dataset=SegmentationDatasetCreator(folder_path,val_img,extractor)
test_dataset=SegmentationDatasetCreator(folder_path,test_img,extractor)

#Number of images at a time
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=workers)
val_dataloader=DataLoader(validation_dataset,batch_size=batch_size,num_workers=workers)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,num_workers=workers)

#id2label dictionary
Label_dictionary= {
    0: "background",
    1: "crater"
}

#This instantiates the finetuner to later pass it onto the trainer
segformer_finetuner = SegformerFinetuner(
    id2label=Label_dictionary, 
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    test_dataloader=test_dataloader, 
    metrics_interval=10,
)

#Stopgap measures
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=3, 
    verbose=False, 
    mode="min",
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

trainer = pl.Trainer(
    accelerator="cpu", 
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=100,
    val_check_interval=len(train_dataloader),
)

#Simple training step. It will take the instructions from segformer_finetuner, which was instantiated already with train, validation and test datasets. 
trainer.fit(segformer_finetuner)

#These next few blocks contain code to show metrics. Note we don't need to call a validation step. The Trainer is doing that at each epoch. All we do is call the model (After training) and pass it the test data. 
#Each metric class is defined so that the model is converted to evaluation mode. 
#%% Visual inspection
visualize_predictions(segformer_finetuner, segformer_finetuner.test_dataloader(), num_images=5):
#%% Not as useful when we only have one class, but it can give you something similar to IoU in a graphical form.
plot_confusion_matrix(segformer_finetuner, segformer_finetuner.test_dataloader())
#%% Intersection over Union is how many predicted pixels were in the correct class. range of 0 to 100% 
iou = mean_iou_single_class(segformer_finetuner, segformer_finetuner.test_dataloader())
print(f"Mean IoU: {iou}")
#%% Not sure on this one yet
precision, recall, f1 = precision_recall_f1(model, val_dataloader)
print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
