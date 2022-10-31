#library imports
# from ast import Pass
# import time 
#from tkinter import Y
from cmath import isnan
from operator import ge
from time import pthread_getcpuclockid
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir
import os


import matplotlib.pyplot as plt
#import torch.optim.lr_scheduler as lr_scheduler
import utils
from torchvision import transforms, datasets
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import confusion_matrix
from Few_shot_datasetup import CustomImageDataset, bbImageDataset, unpickle, add_get_labels, TaskSampler, get_subset_classes, add_class

from methods.bd_cspn import BDCSPN
from methods.finetune import Finetune
from methods.matching_networks import MatchingNetworks
from methods.transductive_finetuning import TransductiveFinetuning
from methods.tim import TIM
from methods.prototypical_networks import PrototypicalNetworks
from BB_model import BB_model,backbone_network,CNN,sigmoid_bb
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import wandb


try:
    from torchvision.ops.giou_loss import generalized_box_iou_loss
except:
    print('latest pytorch version not available')
    from utils import generalized_box_iou_loss

#for running in colab:
try:
    import pickle5 as pickle
except:
    pass

#TODO add dropout
#TODO try training without pre-training
#TODO add training RPN with incremental learning 
#TODO use pretrained inception module 
#
# import  timm
# m = timm.create_model('inception_v4', pretrained=True)
# m.eval()

print('last updated at 15:44')

wandb.login(key='0574c08b0f27afb00c2c12fa83635ea5ffcf823f')
wandb.init(project='my-test-project')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default='FGVCAircraft', help="which dataset to load rs = road_signs cf100 = CIFAR-100 ")
parser.add_argument("-test", "--test", default='n', help="whether to run in test mode or not, y or n")
parser.add_argument("-tr", "--train", default='n', help="whether to train the model or not, y or n")
parser.add_argument("-m", "--model", default='resnet18', help="model type")
parser.add_argument("-l", "--load", default='n', help="whether to load model from file, y or n")
parser.add_argument("-p", "--path", default='road_signs/Models/', help="path to models")
parser.add_argument("-v", "--visualise", default='n', help="whether to visualise output y or n")
parser.add_argument("-exp", "--experiment", default='n', help="run experimental code")
parser.add_argument("-Augment", "--Augment", default='y', help="whether to use autoaugment or not")
parser.add_argument("-fgl", "--fine_grained_loss", default='n', help="whether to use fine grained loss function or not from https://arxiv.org/pdf/1705.08016.pdf")
parser.add_argument("--input_folder",default='n',help='dataset folder for industry data')
parser.add_argument("-new_class","--new_class",default='n',help='id of new class to be added')
parser.add_argument("-nac","--n_add_classes",default='0',help='number of new classes to be added')
parser.add_argument("-wandb","--wandb_run_path",default='n',help='path of previous run to load model from')
parser.add_argument("-f_fgl","--feature_loss",default='n',help='whether to use fine grained loss between features or not')
parser.add_argument("-holes","--holes",default='n',help='whether to use holes in dataset or not')



#bounding box arguments
parser.add_argument("-bb_train", "--bb_train", default='n', help="whether to train bounding box model or not")
parser.add_argument("-bb_test", "--bb_test", default='n', help="whether to test bounding box model or not")
parser.add_argument("-bb_mt", "--bb_model_type", default='bbresnet18', help="name of bb model to be loaded/saved")
parser.add_argument("-bb_crop", "--bb_crop", default='n', help="whether to crop based on bounding boxes or not before classification, [y,n,gt] gt = use ground truth boxes, if y must have previously trained bb model")
parser.add_argument("-bb_model_name", "--bb_model_name", default='bb_default.pt', help="name of bb model to be loaded/saved")
parser.add_argument("-augment_crops", "--augment_crops", default='n', help="whether to add augmentations to after cropping")
parser.add_argument("-lf","--loss_function",default='l1',help='loss function to use for bb prediction')
parser.add_argument("-bb_resize","--bb_resize",default='1',help='size to increase bounding box by')

#few shot arguments
parser.add_argument("-il", "--incremental_learning", default='n', help="whether to train in incremental setting or not")
parser.add_argument("-fsm", "--few_shot_model", default='n', help="which few shot model to run with")
parser.add_argument("-fsft","--fsft", default='n', help="few shot or fine tune: [fs or ft]")
parser.add_argument("-ros","--ros", default='n', help="whether to use random over sampling or not")
parser.add_argument("-rus","--rus", default='n', help="whether to use random under sampling or not")
parser.add_argument("-subsample","--subsample", default='1', help="how much to restrict training set by [0-1]")
parser.add_argument("-invert_fgl","--invert_fgl", default='n', help="whether to invert fgl or not")
parser.add_argument("-unseen_test","--unseen_test", default='n', help="whether to use distinct classes for test set or not")
parser.add_argument("-n_way","--n_way", default='5', help="n_way if performing few shot learning on unseen classes")
parser.add_argument("-n_shot", "--n_shot", default='5', help="number of support images in ")
parser.add_argument("-n_query", "--n_query", default='15', help="number of support images in ")
parser.add_argument("-fs_ds", "--fs_ds", default='n', help="whether to use few shot dataset or not")


#model hyperparameters
parser.add_argument("-epochs", "--epochs", default='50', help="number of epochs to run with")
parser.add_argument("-patience", "--patience", default='10', help="number of epochs to wait for better validation performance")
parser.add_argument("-lr", "--lr", default='0.006', help="learning rate for adam optimiser")
parser.add_argument("-n_base", "--n_base", default='100', help="number of base classes")
parser.add_argument("-n_il_steps", "--n_il_steps", default='10', help="number incremental learning steps")
parser.add_argument("-lambda", "--lambda", default='0', help="hyperparameter for fine grained loss function, (will be scaled by number of classes/100)")
parser.add_argument("-bs", "--batch_size", default='32', help="hyperparameter for fine grained loss function, (will be scaled by number of classes/100)")

args = vars(parser.parse_args())

#convert number inputs to integer
for key in ['n_query','n_way','epochs','patience','n_shot','n_il_steps','n_base','lambda','batch_size','n_add_classes']:
    args[key] = int(args[key])
#convert float inputs to float
for key in ['subsample','lr','bb_resize']:
    args[key] = float(args[key])

#log performance
wandb.config.update(args)
print('performance tracking using #wandb')

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def val_metrics_bb(bb_model, val_dl,base_classes=None, lf="l1"):
    bb_model.eval()
    bb_model.to(device)
    total = 0
    sum_loss = 0
    sum_iou = 0
    if lf == "mse":
        loss_fn = F.mse_loss
    elif lf == "l1":
        loss_fn = F.l1_loss
    elif lf == "iou":
        loss_fn = generalized_box_iou_loss

    for x, y_class, y_bb in tqdm(val_dl):
        if base_classes is not None:
            #get indicies of base classes to use as targets
            y_class = tensor([base_classes.tolist().index(i) for i in y_class])
        x = x.to(device).float()
        y_bb = y_bb.to(device).float()
        out_bb = bb_model(x)
        #loss_bb = F.l1loss(out_bb, y_bb, reduction="none").sum(1)
        loss = loss_fn(out_bb, y_bb,reduction="sum")
        #multiply by boolean if only predicting one
        for i, y_bb_i in enumerate(y_bb):
            iou = utils.calc_IOU(y_bb_i.int().cpu(),out_bb[i].detach().int().cpu(),x[i].cpu())
            if np.isnan(iou):
                iou = 0
                print(f'invalid value encountered, pred_bb was {out_bb[i].int()}, label was {y_bb_i.int()}')
            sum_iou += iou
        sum_loss += loss.item()
        total += y_class.shape[0]
    return sum_loss/total, sum_iou/total

def train_bb_predictor(bb_model, optimizer, train_dl, val_dl, epochs=10, base_classes=None,patience=5, lf="l1",augment_pol=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bb_model.to(device)
    best_val_iou = 0
    bad_epochs = 0 
    if lf == "mse":
        loss_fn = F.mse_loss
    elif lf == "l1":
        loss_fn = F.l1_loss
    elif lf == "iou":
        loss_fn = generalized_box_iou_loss

    for i in range(epochs):
        bb_model.train()
        wandb.watch(bb_model)
        total = 0
        sum_loss = 0

        print(f'epoch {i} of {epochs}')
        for x, y_class, bbs in tqdm(train_dl):
            if base_classes is not None:
                #get indicies of base classes to use as targets
                y_class = tensor([base_classes.tolist().index(i) for i in y_class])
            if augment_pol:
                x,bbs = utils.augment_bbs(x,bbs,augment_pol)
            x = x.to(device).float()
            bbs = bbs.to(device).float()
            bb_preds = bb_model(x)
            loss = loss_fn(bb_preds,bbs, reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += y_class.shape[0]
            sum_loss += loss.item()
        train_loss = sum_loss/total
        print('epoch finished')
                
        val_loss, val_iou = val_metrics_bb(bb_model, val_dl,base_classes,lf)

        print("train_loss %.4f val_loss %.4f val_iou %.4f " % (train_loss,val_loss, val_iou))
        wandb.log({"train_loss": train_loss,"val_loss": val_loss,"val_iou": val_iou})
        

        if val_iou>=best_val_iou:
            bad_epochs = 0
            best_val_iou = val_iou
            print(f'new best model, saved as {os.path.join(wandb.run.dir, "bb_model_dict.pt")}')
            torch.save(bb_model.state_dict(),(os.path.join(wandb.run.dir, "bb_model_dict.pt")))
        else:
            bad_epochs +=1
            print(f'{bad_epochs} of {patience} epochs without improvement')
            if bad_epochs>patience:
                print('finishing training due to max validation performance reached')
                break
    wandb.log({"best_val_iou":best_val_iou})

    return sum_loss/total


def val_metrics_classifier(model, val_dl,base_classes = None,bb_crop='n',bb_model=None, feature_output =False):
    """function to test classifier performance
    Args:
        model: pytorch model to be tested 
        val_dl: data loader with validation/test set
        base_classes: list/array of subsampled classes
    Returns:
        Average loss
        Average accuracy
        Class accuracies (in same order as base classes) """
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    ground_truths = []
    pred_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if bb_crop=='y':
        bb_model.to(device)

    for x, y_class, bbs in tqdm(val_dl):
        if base_classes is not None:
            #get indicies of base classes to use as targets
            y_class = tensor([base_classes.tolist().index(i) for i in y_class])
        with torch.no_grad():
            x = x.to(device).float()
            if bb_crop =='y':
                bb_preds = bb_model(x)
                try:
                    x = utils.bb_crop_resize(x,bb_preds,width=256,height=256).to(device)
                except:
                    print('error during re-sizing, skipping batch')
                    continue
            elif bb_crop =='gt':
                try:
                    x = utils.bb_crop_resize(x,bbs,width=256,height=256).to(device)
                except:
                    print('error during re-sizing, skipping batch')
                    continue
            y_class = y_class.to(device)
            batch = y_class.shape[0]
            if feature_output:
                out_class,_ = model(x)
            else:
                out_class = model(x)
            loss = F.cross_entropy(out_class, y_class, reduction="sum")
            #multiply by boolean if only predicting one
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(y_class).sum().item()
            sum_loss += loss.item()
            total += batch
            #append to lists
            ground_truths+= (y_class.tolist())
            pred_list +=(pred.tolist())

    matrix = confusion_matrix(ground_truths, pred_list)
    #in same order as base_classes
    class_accuracies = matrix.diagonal()/matrix.sum(axis=1)
        
    return sum_loss/total, correct/total, class_accuracies, ground_truths, pred_list

def val_metrics_few_shot(model, val_dl,bb_crop='n',bb_model=None, feature_output =False):
    """function to test few_shot classifier performance on unseen classes
    Args:
        model: pytorch model to be tested 
        val_dl: data loader with validation/test set
        base_classes: list/array of subsampled classes
    Returns:
        Average loss
        Average accuracy
        Class accuracies (in same order as base classes)
        ground_truths
        pred_list """
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    ground_truths = []
    pred_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if bb_crop=='y':
        bb_model.to(device)
    for sup_ims, sup_labs, sup_bbs, q_ims, q_labs, q_bbs, class_ids in tqdm(val_dl):
        with torch.no_grad():
            sup_ims = sup_ims.to(device).float()
            q_ims = q_ims.to(device).float()
            sup_labs = sup_labs.to(device)
            q_labs = q_labs.to(device)
            if bb_crop =='y':
                sup_bb_preds = bb_model(sup_ims)
                q_bb_preds = bb_model(q_ims)
                try:
                    sup_ims = utils.bb_crop_resize(sup_ims,sup_bb_preds,width=256,height=256).to(device)
                    q_ims = utils.bb_crop_resize(q_ims,q_bb_preds,width=256,height=256).to(device)
                except:
                    print('error during re-sizing, skipping batch')
                    continue
            elif bb_crop =='gt':
                try:
                    sup_ims = utils.bb_crop_resize(sup_ims,sup_bbs,width=256,height=256).to(device)
                    q_ims = utils.bb_crop_resize(q_ims,q_bbs,width=256,height=256).to(device)
                except:
                    print('error during re-sizing, skipping batch')
                    continue
            model.process_support_set(sup_ims,sup_labs)
            batch = q_labs.shape[0]
            if feature_output:
                out_class,_ = model(q_ims)
            else:
                out_class = model(q_ims)
            loss = F.cross_entropy(out_class, q_labs, reduction="sum")
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(q_labs).sum().item()
            sum_loss += loss.item()
            total += batch
            #append to lists
            ground_truths+= list(class_ids[q_labs.cpu()])
            pred_list +=list(class_ids[pred.cpu()])

    matrix = confusion_matrix(ground_truths, pred_list)
    #in same order as base_classes
    class_accuracies = matrix.diagonal()/matrix.sum(axis=1)
        
    return sum_loss/total, correct/total, class_accuracies, ground_truths, pred_list



def train_classifier_feature_fgl(model, optimizer, train_dl, val_dl, epochs, patience,
 base_classes=None,bb_crop = None,bb_model=None,fgl='n',lambda_=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if bb_crop=='y':
        bb_model.to(device)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    bad_epochs = 0 
    batch_size = train_dl.batch_size
    half_batch_size = int(batch_size/2)
    if fgl=='y':
        assert batch_size%2==0,f"batch size must be even number, size was {batch_size}"

    for i in range(epochs):
        model.train()
        wandb.watch(model)
        total = 0
        sum_loss = 0
        correct = 0
        print(f'epoch {i} of {epochs}')
        for x, y_class, bbs in tqdm(train_dl):
            x = x.to(device).float()
            bbs = bbs.to(device)
            if base_classes is not None:
                #get indicies of base classes to use as targets
                y_class = tensor([base_classes.tolist().index(i) for i in y_class])
            y_class = y_class.to(device)
            total += y_class.shape[0]
            if bb_crop =='y':
                with torch.no_grad():
                    bb_preds = bb_model(x)
                    try:
                        x = utils.bb_crop_resize(x,bb_preds,width=256,height=256)
                    except:
                        print('error during re-sizing, skipping batch')
                        continue
    
            elif bb_crop =='gt':
                with torch.no_grad():
                    try:
                        x = utils.bb_crop_resize(x,bbs,width=256,height=256)
                    except:
                        print('error during re-sizing, skipping batch')
                        continue

            if args['augment_crops']=='y':
                if args['dataset']=='industry':
                    x = transforms.Compose([transforms.RandomRotation(360,fill=0.5),transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),transforms.Resize(size=(256,256))])(x)
                else:
                    x = transforms.AutoAugment()((x*256).to(dtype=torch.uint8))/256

            x = x.to(device).float()
            out_class, features = model(x)
            y_class = y_class.to(device)
            loss_CE = F.cross_entropy(out_class, y_class, reduction="sum")
            if fgl == 'y':
            #pairwise confusion loss function from https://arxiv.org/pdf/1705.08016.pdf             
                gamma = (y_class[:half_batch_size]==y_class[half_batch_size:])
                D_EC = F.mse_loss(features[:half_batch_size]*gamma[:,None],features[half_batch_size:]*gamma[:,None])
                loss = loss_CE + lambda_*D_EC
                print(f'loss_CE is {loss_CE}, lambda_*D_EC is {lambda_*D_EC}')
            else:
                loss = loss_CE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(y_class).sum().item()

            sum_loss += loss.item()
        train_loss = sum_loss/total
        train_acc = correct/total
        print('epoch finished')
                
        val_loss, val_acc, _, _, _ = val_metrics_classifier(model, val_dl,base_classes,bb_crop,bb_model,feature_output=True)
        #scheduler.step(val_loss)
        #print(f' learning rate is {get_lr(optimizer)}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("train_loss %.4f train_acc %.4f val_loss %.4f val_acc %.4f " % (train_loss,train_acc, val_loss, val_acc))
        wandb.log({"train_loss": train_loss,"train_acc": train_acc,"val_loss": val_loss,"val_acc": val_acc})
        

        if val_acc>=best_val_acc:
            bad_epochs = 0
            best_val_acc = val_acc
            print(f'new best model, saved as {os.path.join(wandb.run.dir, "model_dict.pt")}')
            torch.save(model.state_dict(),(os.path.join(wandb.run.dir, "model_dict.pt")))
        else:
            bad_epochs +=1
            print(f'{bad_epochs} of {patience} epochs without improvement')
            if bad_epochs>patience:
                print('finishing training due to max validation performance reached')
                break
    wandb.log({"best_val_acc":best_val_acc})

    return sum_loss/total


def train_classifier_with_bb_crop_and_fgl(model, optimizer, train_dl, val_dl, epochs, patience,
 base_classes=None,bb_crop = None,bb_model=None,fgl='n',lambda_=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if bb_crop=='y':
        bb_model.to(device)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    bad_epochs = 0 
    batch_size = train_dl.batch_size
    half_batch_size = int(batch_size/2)
    if fgl=='y':
        assert batch_size%2==0,f"batch size must be even number, size was {batch_size}"

    for i in range(epochs):
        model.train()
        wandb.watch(model)
        total = 0
        sum_loss = 0
        correct = 0
        print(f'epoch {i} of {epochs}')
        for x, y_class, bbs in tqdm(train_dl):
            x = x.to(device).float()
            bbs = bbs.to(device)
            if base_classes is not None:
                #get indicies of base classes to use as targets
                y_class = tensor([base_classes.tolist().index(i) for i in y_class])
            y_class = y_class.to(device)
            total += y_class.shape[0]
            if bb_crop =='y':
                with torch.no_grad():
                    bb_preds = bb_model(x)
                    try:
                        x = utils.bb_crop_resize(x,bb_preds,width=256,height=256)
                    except:
                        print('error during re-sizing, skipping batch')
                        continue
    
            elif bb_crop =='gt':
                with torch.no_grad():
                    try:
                        x = utils.bb_crop_resize(x,bbs,width=256,height=256)
                    except:
                        print('error during re-sizing, skipping batch')
                        continue

            if args['augment_crops']=='y':
                if args['dataset']=='industry':
                    x = transforms.Compose([transforms.RandomRotation(360,fill=0.5),transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),transforms.Resize(size=(256,256))])(x)
                else:
                    x = transforms.AutoAugment()((x*256).to(dtype=torch.uint8))/256

            x = x.to(device).float()
            out_class = model(x).to(device)
            y_class = y_class.to(device)
            loss_CE = F.cross_entropy(out_class, y_class, reduction="sum")
            if fgl == 'y':
            #pairwise confusion loss function from https://arxiv.org/pdf/1705.08016.pdf 
                if args['invert_fgl']=='y':            
                    gamma = (y_class[:half_batch_size]==y_class[half_batch_size:])
                else:
                    gamma = (y_class[:half_batch_size]!=y_class[half_batch_size:])
                D_EC = F.mse_loss(out_class[:half_batch_size]*gamma[:,None],out_class[half_batch_size:]*gamma[:,None])
                print(f'euclidean loss is {lambda_*D_EC}, classification loss is {loss_CE}')
                loss = loss_CE + lambda_*D_EC
            else:
                loss = loss_CE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out_class, 1)
            correct += pred.eq(y_class).sum().item()

            sum_loss += loss.item()
        train_loss = sum_loss/total
        train_acc = correct/total
        print('epoch finished')
                
        val_loss, val_acc, _, _, _ = val_metrics_classifier(model, val_dl,base_classes,bb_crop,bb_model)
        #scheduler.step(val_loss)
        #print(f' learning rate is {get_lr(optimizer)}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("train_loss %.4f train_acc %.4f val_loss %.4f val_acc %.4f " % (train_loss,train_acc, val_loss, val_acc))
        wandb.log({"train_loss": train_loss,"train_acc": train_acc,"val_loss": val_loss,"val_acc": val_acc})
        

        if val_acc>=best_val_acc:
            bad_epochs = 0
            best_val_acc = val_acc
            print(f'new best model, saved as {os.path.join(wandb.run.dir, "model_dict.pt")}')
            torch.save(model.state_dict(),(os.path.join(wandb.run.dir, "model_dict.pt")))
        else:
            bad_epochs +=1
            print(f'{bad_epochs} of {patience} epochs without improvement')
            if bad_epochs>patience:
                print('finishing training due to max validation performance reached')
                break
    wandb.log({"best_val_acc":best_val_acc})

    return sum_loss/total


if __name__ == '__main__':
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device}')

############ LOAD SPECIFIED DATASET ############
################################################


########## INDUSTRY DATASET #################I

    if args['dataset']=='industry':
        #load datasets
        img_dir = args['input_folder']+'/images/'
        label_dir = args['input_folder']+'/labels/'
        if args['Augment']=='y' and args['bb_train']=='y':

            # im_transforms = transforms.RandomApply(torch.nn.ModuleList([
            # transforms.GaussianBlur(kernel_size=(1,11),sigma=(0.1,5))
            # ,transforms.RandomAdjustSharpness(10,p=0.5)
            # ,transforms.ColorJitter(brightness=0.2,saturation=1,contrast=0.5,hue=0.1)
            # ]),p=0.7)
            trainset = utils.industry_dataset(label_dir=label_dir+'train/',img_dir=img_dir+'train/',transforms=True,margin=args['bb_resize'],im_transforms=None)
        else:
            trainset = utils.industry_dataset(label_dir=label_dir+'train/',img_dir=img_dir+'train/',transforms=False,margin=args['bb_resize'])
        valset = utils.industry_dataset(label_dir=label_dir+'valid/',img_dir=img_dir+'valid/',transforms=False,margin=args['bb_resize'])
        testset = utils.industry_dataset(label_dir=label_dir+'test/',img_dir=img_dir+'test/',transforms=False,margin=args['bb_resize'])
        print(f'loaded datasets, trainset has {trainset.__len__()} datapoints, valset: {valset.__len__()}, testset: {testset.__len__()}')

        #specify classes
        if args['holes'] =='y':
            all_classes = np.array(['23', '02', '12', '4', '0', '2', '03',''])
            class_names =['thread_washer','nut_thread','spacer_thread','socket_bolt','nut','thread','nut_washer','holes']
        else:
            all_classes = np.array(['23', '02', '12', '4', '0', '2', '03'])
            class_names =['thread_washer','nut_thread','spacer_thread','socket_bolt','nut','thread','nut_washer']
        original_class_names = class_names
        class_name_dict = {}
        for i, class_id in enumerate(all_classes):
            class_name_dict[class_id] = class_names[i]

        #load classes from previous run
        if args['wandb_run_path']!='n' and args['holes'] =='n':
            api = wandb.Api()
            run = api.run(args['wandb_run_path'])
            base_classes = np.array(run.config['base_classes'])
            print('base classes loaded from previous run')
        elif args['n_base']<len(all_classes):
            base_classes = np.random.choice(all_classes,args['n_base'],replace=False)
        else:
            base_classes = all_classes

        class_names = [class_name for i,class_name in enumerate(class_names) if all_classes[i] in base_classes]
        wandb.config.base_classes = base_classes
        print(f'base_classes are {base_classes}')
        
        trainset = utils.subsetclasses(trainset,trainset._labels,base_classes)
        valset = utils.subsetclasses(valset,valset._labels,base_classes)
        testset = utils.subsetclasses(testset,testset._labels,base_classes)
        print(f'After removing holes and mislabelled data: trainset has {trainset.__len__()} datapoints, valset: {valset.__len__()}, testset: {testset.__len__()}')

        batch_size = args['batch_size']
        #restrict number of data points, in training set. balance by under or over sampling as specified
        if args['rus']=='y' or args['new_class']!='n':
            #ensure reproducibility, take same subsample of data
            if args['rus']=='y':
                if args['new_class']!= 'n':
                    few_shot_class = args['new_class']
                else:
                    few_shot_class = np.random.choice(all_classes)
                    print(f'few shot class is {few_shot_class}')
                main_classes = np.array([i for i in all_classes if i != few_shot_class])
                trainset=utils.subsetter(trainset,trainset._labels,main_classes,few_shot_class,args['n_shot'],random_seed=0,data_frac=args['subsample'])
        elif args['subsample'] <1:
            g = torch.Generator()
            g.manual_seed(0)
            train_len = trainset.__len__()
            split_len = int(args['subsample']*train_len)
            trainset, _ = utils.custom_random_split(trainset,trainset._labels,[split_len,train_len-split_len],generator=g)
        elif args['fs_ds']=='y':
            trainset = utils.create_few_shot_subset(trainset,args['n_way'],args['n_shot'])
            valset = trainset
            testset = utils.subsetclasses(testset,testset._labels,trainset._base_classes)
            base_classes = trainset._base_classes


        if args['ros']=='y':
            train_weights = utils.check_class_balance(trainset._labels,plot=False)
            #returns inverse of count of each label, for each index, which is proportional to the sample probability, = Random Over Sampling
            train_weight_vals = tensor([1./train_weights[i] for i in trainset._labels])
            train_dl = DataLoader(trainset, batch_size=batch_size,drop_last=True, sampler=WeightedRandomSampler(train_weight_vals,num_samples=trainset.__len__()))
        else: 
            train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True)
        
        #balance validation set if specified
        if args['ros']=='y'or args['rus']=='y':
            val_weights = utils.check_class_balance(valset._labels,plot=False)
            val_weight_vals = tensor([1./val_weights[i] for i in valset._labels])
            val_dl = DataLoader(valset, batch_size=batch_size,drop_last=True, sampler=WeightedRandomSampler(val_weight_vals,num_samples=valset.__len__()))
        else:
            val_dl = DataLoader(valset, batch_size=batch_size,drop_last=False)

        test_dl = DataLoader(testset, batch_size=batch_size,drop_last=False)

        class_balance = pd.DataFrame(utils.check_class_balance(trainset._labels,plot=False),index=[0])
        wandb.config.class_balance = class_balance

########## FGVC Aircraft DATASET #################

    elif args['dataset']=='FGVCAircraft':
        with open('FGVCAircraft/fgvc-aircraft-2013b/data/variants.txt') as f:
            class_names = f.readlines()
        print(f'class names are {class_names}')
        
        class_name_dict = class_names
        #define transforms, incompatible with bb_prediction
        train_transform = transforms.Compose([
               transforms.ToTensor()
             ])
        im_transform = train_transform

        trainset = datasets.FGVCAircraft(root='FGVCAircraft', split='trainval',download=False, transform=train_transform)
        testset = datasets.FGVCAircraft(root='FGVCAircraft', split='test',download=False, transform = im_transform)

        if args['Augment']=='y' and args['bb_train']=='y':
            trainset = utils.bb_dataset(trainset,trainset._image_files,transforms=True)
        else:
            trainset = utils.bb_dataset(trainset,trainset._image_files,transforms=False)
        testset = utils.bb_dataset(testset,testset._image_files,transforms=False)
        
        all_classes = np.unique(trainset._labels)
        print(f'trainset length is {trainset.__len__()}')
        #select subset of classes
        n_classes = len(class_names)

        if args['load']=='n':
            base_classes = np.random.choice(n_classes,args['n_base'],replace=False)
            print('new base classes generated')
        else:
            api = wandb.Api()
            run = api.run(args['wandb_run_path'])
            base_classes = run.config['base_classes']
            #convert formatting
            base_class_str = ''
            for i in base_classes:
                base_class_str += i
            base_classes = base_class_str.strip('[]').split()
            base_classes = np.array([int(i) for i in base_classes])
            print('base classes from classifier model loaded')

        #save subset classes
        print(f'base_classes are {base_classes}')
        wandb.config.base_classes = base_classes

        train_len = trainset.__len__()
        split_len = int(0.8*train_len)
        g = torch.Generator()
        g.manual_seed(0)
        trainset, valset = utils.custom_random_split(trainset,trainset._labels,[split_len,train_len-split_len],generator=g)

        if args['fs_ds']=='y':
            trainset = utils.create_few_shot_subset(trainset,args['n_way'],args['n_shot'])
            valset = trainset
            testset = utils.subsetclasses(testset,testset._labels,trainset._base_classes)
            base_classes = trainset._base_classes
        else:
            trainset = utils.subsetclasses(trainset,trainset._labels,base_classes)
            valset = utils.subsetclasses(valset,valset._labels,base_classes)

        if args['unseen_test'] == 'n':
            testset = utils.subsetclasses(testset,testset._labels,base_classes)
        
        class_names = [class_names[i] for i in base_classes]
        print(f'datasets filtered to only contain {len(class_names)} classes')
        print(f'subset classes selected, trainset length is {trainset.__len__()}')

        print('split into val and trainset')
        print(f'trainset length is {trainset.__len__()}')
        print(f'valset length is {valset.__len__()}')
        #update datasets to only contain subset classes

        batch_size = args['batch_size']
        train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True)
        val_dl = DataLoader(valset, batch_size=batch_size,drop_last=False)
        test_dl = DataLoader(testset, batch_size=batch_size,drop_last=False)

############ ADD NEW CLASSES #########################
######################################################

    if args['n_add_classes']>0 or args['unseen_test'] == 'y' :
        #TODO add a way to load additional classes from previous run
        n_add_classes = args['n_add_classes']
        main_classes = base_classes
        if args['load']=='y':
            api = wandb.Api()
            run = api.run(args['wandb_run_path'])
            few_shot_classes = np.array(run.config['few_shot_classes'])

            print(f'few_shot_classes loaded from previous run are {few_shot_classes}')
        else:
            available_classes = np.array([i for i in all_classes if i not in base_classes])
            few_shot_classes = np.random.choice(available_classes,args['n_add_classes'],replace=False)
            print(f'few_shot_classes generated are {few_shot_classes}')
            wandb.config.few_shot_classes = few_shot_classes

        trainset = utils.add_classes_to_dataset(trainset,n_add_classes,args['n_shot'],new_classes=few_shot_classes)
        valset = utils.add_classes_to_dataset(valset,n_add_classes,np.inf,new_classes=few_shot_classes)

        #base_classes = all classes in training set
        base_classes = trainset._base_classes
        print(f'base classes are {base_classes}')

        if args['unseen_test'] == 'n':
            #test on same classes to training
            testset = utils.add_classes_to_dataset(testset,n_add_classes,np.inf,new_classes=few_shot_classes)
            class_names = [class_name_dict[i] for i in base_classes]
        elif args['unseen_test'] == 'y' and args['train']=='y': 
            #test on different classes to training
            un_seen_classes = [i for i in all_classes if i not in base_classes]
            testset = utils.subsetclasses(testset,testset._labels,un_seen_classes)
            class_names = [class_name_dict[i] for i in un_seen_classes]
        elif args['unseen_test'] =='y' and args['train']=='n':
            un_seen_classes = base_classes

        if args['ros']=='y':
            train_weights = utils.check_class_balance(trainset._labels,plot=False)
            #returns inverse of count of each label, for each index, which is proportional to the sample probability, = Random Over Sampling
            train_weight_vals = tensor([1./train_weights[i] for i in trainset._labels])
            train_dl = DataLoader(trainset, batch_size=batch_size,drop_last=True, sampler=WeightedRandomSampler(train_weight_vals,num_samples=trainset.__len__()))
        else: 
            train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True)
        
        #balance validation set if specified
        if args['ros']=='y'or args['rus']=='y':
            val_weights = utils.check_class_balance(valset._labels,plot=False)
            val_weight_vals = tensor([1./val_weights[i] for i in valset._labels])
            val_dl = DataLoader(valset, batch_size=batch_size,drop_last=True, sampler=WeightedRandomSampler(val_weight_vals,num_samples=valset.__len__()))
        else:
            val_dl = DataLoader(valset, batch_size=batch_size,drop_last=False)

        #set up traditional few shot test set on unseen classes
        if args['unseen_test'] == 'y' and args['few_shot_model']!= 'n':
            n_tasks = np.ceil(testset.__len__()/((args['n_query']+args['n_shot'])*args['n_way'])).astype(int)
            bb_test_sampler = utils.bb_TaskSampler(
                testset, n_way=args['n_way'], n_shot=args['n_shot'], n_query=args['n_query'], n_tasks=n_tasks
            )
            test_dl = DataLoader(testset,batch_sampler=bb_test_sampler,num_workers=2,
                pin_memory=True,collate_fn=bb_test_sampler.episodic_collate_fn
            )

        else:
            test_dl = DataLoader(testset, batch_size=batch_size,drop_last=False)

        class_balance = pd.DataFrame(utils.check_class_balance(trainset._labels,plot=False),index=[0])
        wandb.config.class_balance = class_balance

############# DEFINE MODEL TYPE ##################
###################################################

    if args['feature_loss']=='y':
        output_features = True
    else:
        output_features = False

    if args['model']=='BB_model':
        model = BB_model().to(device)

    elif args['model']=='resnet12':
        from easyfsl.modules import resnet12
        n_classes = len(base_classes)

        model = resnet12(
        use_fc=True,
        num_classes=n_classes,
        ).to(device)
        print('resnet12 model loaded')

    elif args['model']=='resnet18':

        pre_trainined_model = models.resnet18(weights='IMAGENET1K_V1')
        n_classes = len(base_classes)
        model = backbone_network(n_classes,pre_trainined_model,output_features = output_features)
        model.to(device)
        print('resnet model loaded')

    elif args['model']=='resnet18_ri':
        #uses random initialisation
        pre_trainined_model = models.resnet18()
        n_classes = len(base_classes)
        model = backbone_network(n_classes,pre_trainined_model,output_features = output_features)
        model.to(device)
        print('resnet model with no pre-training loaded')

    elif args['model']=='simple_cnn':
        n_classes = len(base_classes)
        model = CNN(n_classes)
        model.to(device)
        print('simple CNN model loaded')

    elif args['model']=='finetune_resnet18':
        n_classes = len(base_classes)
        model = backbone_network(n_classes,fixed_backbone=True,use_fc_layer=True)
        model.to(device)
        print('pretrained resnet, fixed backbone model loaded')
    
    if args['bb_model_type']=='bbresnet18':
        pre_trainined_model = models.resnet18(weights='IMAGENET1K_V1')
        bb_model = backbone_network(4,pre_trainined_model)
        bb_model.to(device)
        print('bbresnet18 model loaded')

    elif args['bb_model_type']=='bbresnet18_ri':
        pre_trainined_model = models.resnet18()
        bb_model = backbone_network(4,pre_trainined_model)
        bb_model.to(device)
        print('bbresnet18 model loaded')

    elif args['bb_model_type']=='sigmoid_bb':
        pre_trainined_model = models.resnet18()
        bb_model = sigmoid_bb(pre_trainined_model)
        bb_model.to(device)
        print('sigmoid bb model loaded')

############ LOAD PREVIOUSLY SAVED MODEL ##################
###########################################################

    if args['bb_crop']=='y':

        state_dict_file = wandb.restore("bb_model_dict.pt",run_path=args['wandb_run_path'])
        state_dict = torch.load(state_dict_file.name).copy()    
        bb_model.load_state_dict(state_dict)
        bb_model.eval()
        bb_model.to(device)
        print('bb model loaded')

    if args['load']=='y':
        if args['bb_train']=='y':
            state_dict_file = wandb.restore("bb_model_dict.pt",run_path=args['wandb_run_path'])
            state_dict = torch.load(state_dict_file.name).copy()    
            bb_model.load_state_dict(state_dict)
            bb_model.eval()
            bb_model.to(device)
            print('bb model loaded')
        else:
            state_dict_file = wandb.restore("model_dict.pt",run_path=args['wandb_run_path'])
            state_dict = torch.load(state_dict_file.name).copy()   
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            print('classifier model loaded')

############### TRAIN MODEL #####################
#################################################

    if args['bb_train']=='y':
        print('training model')
        parameters = filter(lambda p: p.requires_grad, bb_model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args['lr'])
        if args['Augment']=='y':
            augment_pol = transforms.Compose([
                transforms.RandomHorizontalFlip()
                ,transforms.RandomVerticalFlip()
                ,transforms.RandomApply([transforms.GaussianBlur(kernel_size=11,sigma=(0.5,5)),transforms.RandomAdjustSharpness(5,p=1)])
                ,transforms.RandomRotation(360)
                ])
        else:
            augment_pol = None
        _ = train_bb_predictor(bb_model, optimizer , train_dl, val_dl, epochs=args['epochs'], base_classes=base_classes, patience=args['patience'],lf=args['loss_function'],augment_pol=augment_pol)
       
    if args['train']=='y':
        print('training model')
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args['lr'])

        #hyperparameter will become more potent for lower number of classes, so rescale to original number of classes
        lambda_ = args['lambda']*n_classes/100 
        if args['feature_loss'] =='y':
            _ = train_classifier_feature_fgl(model, optimizer , train_dl, val_dl, epochs=args['epochs']
             , base_classes=base_classes, patience=args['patience'],bb_crop = args['bb_crop'],bb_model=bb_model
             ,fgl=args['fine_grained_loss'],lambda_=lambda_)
        else:
            _ = train_classifier_with_bb_crop_and_fgl(model, optimizer , train_dl, val_dl, epochs=args['epochs']
             , base_classes=base_classes, patience=args['patience'],bb_crop = args['bb_crop'],bb_model=bb_model
             ,fgl=args['fine_grained_loss'],lambda_=lambda_)

################# SET UP FEW SHOT MODEL ####################
############################################################

    if args['fsft']=='fs':

        if args['train']=='y':
            #load file just generated 
            state_dict_file = os.path.join(wandb.run.dir, "model_dict.pt")
            state_dict = torch.load(state_dict_file)
        elif args['load']=='y':
            #load previously created file
            state_dict_file = wandb.restore("model_dict.pt",run_path=args['wandb_run_path'])
            state_dict = torch.load(state_dict_file.name).copy()

        model.set_use_fc(False)

        if args['few_shot_model']=='prototypical':
            model = PrototypicalNetworks(model).to(device)
            print('prototypical few shot model loaded')

        elif args['few_shot_model']=='tim':
            model = TIM(model).to(device)
            print('TIM few shot model loaded')

        elif args['few_shot_model']=='TransductiveFinetuning':
            model = TransductiveFinetuning(model).to(device)
            print('TransductiveFinetuning few shot model loaded')
        
        elif args['few_shot_model']=='MatchingNetworks':
            model = MatchingNetworks(model).to(device)
            print('MatchingNetworks few shot model loaded')

        elif args['few_shot_model']=='Finetune':
            model = Finetune(model).to(device)
            print('Finetune few shot model loaded')

        elif args['few_shot_model']=='BDCSPN':
            model = BDCSPN(model).to(device)
            print('BDCSPN few shot model loaded')


        if args['unseen_test'] =='n':
            label_dict = utils.get_dict_of_labels(trainset._labels)
            support_ims, support_labels, support_bbs = utils.get_sample_images(label_dict,trainset,args['n_shot'],base_classes)
            support_ims = utils.bb_crop_resize(support_ims,support_bbs,width=256,height=256)
            
            print(f'support labels are {support_labels}')

            support_ims = support_ims.to(device)
            support_labels = support_labels.to(device)
            model.process_support_set(support_ims,support_labels)
        torch.save(model.state_dict(),(os.path.join(wandb.run.dir, "model_dict.pt")))

    elif args['fsft']=='ft':
        #class balance

        batch_size = args['batch_size']
        #returns dictionary of count of each class
        train_weights = utils.check_class_balance(trainset._labels,plot=False)
        val_weights = utils.check_class_balance(valset._labels,plot=False)
        #returns inverse of count of each label, for each index, which is proportional to the sample probability, = Random Over Sampling
        train_weight_vals = tensor([1./train_weights[i] for i in trainset._labels])
        val_weight_vals = tensor([1./val_weights[i] for i in valset._labels])

        train_dl = DataLoader(trainset, batch_size=batch_size,drop_last=True, sampler=WeightedRandomSampler(train_weight_vals,num_samples=trainset.__len__()))
        val_dl = DataLoader(valset, batch_size=batch_size,drop_last=True, sampler=WeightedRandomSampler(val_weight_vals,num_samples=valset.__len__()))
        
        #reinitialise and retrain classifier https://arxiv.org/pdf/1910.09217.pdf#page=6&zoom=100,197,268
        model.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, n_classes))
        model.fixed_backbone =True

        print('training classifier')
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args['lr'])

        #hyperparameter will become more potent for lower number of classes, so rescale to original number of classes
        lambda_ = args['lambda']*n_classes/100 

        _ = train_classifier_with_bb_crop_and_fgl(model, optimizer, train_dl, val_dl, epochs=args['epochs']
        , base_classes=base_classes, patience=args['patience'],bb_crop = args['bb_crop'],bb_model=bb_model
        ,fgl=args['fine_grained_loss'],lambda_=lambda_)

        
#################### TEST MODEL #######################
#######################################################

    if args['bb_test']=='y':
        print('testing model')
        try:
            #load file just generated 
            state_dict_file = os.path.join(wandb.run.dir, "bb_model_dict.pt")
            state_dict = torch.load(state_dict_file)
        except:
            #load previously created file
            state_dict_file = wandb.restore("bb_model_dict.pt",run_path=args['wandb_run_path'])
            state_dict = torch.load(state_dict_file.name)
        
        bb_model.load_state_dict(state_dict)
        bb_model.eval()
        bb_model.to(device)
        test_loss, test_iou = val_metrics_bb(bb_model,test_dl,base_classes,lf=args['loss_function'])
        wandb.log({"test_loss":test_loss,"test_iou":test_iou})


    if args['test']=='y':
        print('testing model')
        try:
            #load file just generated 
            state_dict_file = os.path.join(wandb.run.dir, "model_dict.pt")
            state_dict = torch.load(state_dict_file)
        except:
            #load previously created file
            state_dict_file = wandb.restore("model_dict.pt",run_path=args['wandb_run_path'])
            state_dict = torch.load(state_dict_file.name)

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        if args['unseen_test']=='y':
            test_loss, test_acc, test_accuracies, ground_truths, predictions = val_metrics_few_shot(model,test_dl,args['bb_crop'],bb_model)
            #predictions must not be greater than number of classes
            ground_truths = tensor([list(un_seen_classes).index(i) for i in ground_truths])
            predictions = tensor([list(un_seen_classes).index(i) for i in predictions])
            #not all unseen classes are necessarily sampled in task sampler
            sampled_classes = np.unique(ground_truths)
            print(f'sampled_classes were {sampled_classes}')
            class_names = np.array(class_names)[sampled_classes.astype(int)]
            ground_truths = np.array([list(sampled_classes).index(i) for i in ground_truths])
            predictions = np.array([list(sampled_classes).index(i) for i in predictions])
            #for reporting accuracies
            base_classes = sampled_classes

        else:
            test_loss, test_acc, test_accuracies, ground_truths, predictions = val_metrics_classifier(model,test_dl,base_classes,args['bb_crop'],bb_model)

        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truths, preds=predictions,
                        class_names=class_names)})

        #repeat 3 times to bootstrap on test set
        bal_gts, bal_preds = utils.balanace_preds_list(ground_truths,predictions)
        wandb.log({"conf_mat_bal_1" : wandb.plot.confusion_matrix(probs=None,y_true=bal_gts, preds=bal_preds,class_names=class_names)})
        bal_gts, bal_preds = utils.balanace_preds_list(ground_truths,predictions)
        wandb.log({"conf_mat_bal_2" : wandb.plot.confusion_matrix(probs=None,y_true=bal_gts, preds=bal_preds,class_names=class_names)})
        bal_gts, bal_preds = utils.balanace_preds_list(ground_truths,predictions)
        wandb.log({"conf_mat_bal_3" : wandb.plot.confusion_matrix(probs=None,y_true=bal_gts, preds=bal_preds,class_names=class_names)})

        accuracy_dict = {}
        for i,class_ in enumerate(base_classes):
            accuracy_dict[str(class_)] = test_accuracies[i]

        accuracy_df = pd.DataFrame(accuracy_dict,index=[0])
        wandb.log({"class_accuracies":accuracy_df})

        if args['new_class']!='n':
            av_orig_class_acc = np.mean([accuracy_dict[str(class_)] for class_ in main_classes])
            new_class_acc = accuracy_dict[args['new_class']]
            wandb.log({"av_orig_class_acc":av_orig_class_acc,"new_class_acc":new_class_acc})
        elif args['n_add_classes']>0 and args['unseen_test']=='n':
            av_orig_class_acc = np.mean([accuracy_dict[str(class_)] for class_ in main_classes])
            av_new_class_acc = np.mean([accuracy_dict[str(class_)] for class_ in few_shot_classes])
            wandb.log({"av_orig_class_acc":av_orig_class_acc,"av_new_class_acc":av_new_class_acc})

        av_class_acc = np.mean(test_accuracies)
        wandb.log({"test_accuracy":test_acc,"test_loss":test_loss,"av_class_accuracy":av_class_acc})
