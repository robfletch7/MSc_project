import cv2
import numpy as np
import torch
import os
from os import listdir
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, Sampler
from torch import tensor, Tensor, default_generator, randperm
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
from tqdm import tqdm

def get_labels(self):
    return self._labels

Subset.get_labels = get_labels

def generalized_box_iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Original implementation from
    https://github.com/facebookresearch/fvcore/blob/bfff2ef/fvcore/nn/giou_loss.py

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``, and The two boxes should have the
    same dimensions.

    Args:
        boxes1 (Tensor[N, 4] or Tensor[4]): first set of boxes
        boxes2 (Tensor[N, 4] or Tensor[4]): second set of boxes
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            applied to the output. ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Reference:
        Hamid Rezatofighi et. al: Generalized Intersection over Union:
        A Metric and A Loss for Bounding Box Regression:
        https://arxiv.org/abs/1902.09630
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss




class bb_TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data):
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """
    
        true_class_ids = list({x[1] for x in input_data})
        all_bbs = torch.stack(([tensor(x[2]) for x in input_data]))
        print(all_bbs.shape)
        all_bbs = all_bbs.reshape(
            (self.n_way, self.n_shot + self.n_query, 4)
        )

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape((-1, *all_images.shape[2:]))
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))

        support_labels = all_labels[:,:self.n_shot].flatten()
        query_labels = all_labels[:,self.n_shot :].flatten()

        support_bbs = all_bbs[:,:self.n_shot].reshape((-1, 4))
        query_bbs = all_bbs[:,self.n_shot:].reshape((-1, 4))

        return (
            support_images,
            support_labels.long(),
            support_bbs,
            query_images,
            query_labels.long(),
            query_bbs,
            np.array(true_class_ids)
        )



class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        sample_classes = None
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
            sample_classes: (optional), which classes to sample from 
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.sample_classes = sample_classes
        self.items_per_label = {}
        #print('starting getting labels')

        #create dictionary with labels of all indicies, so classes can be sampled seperately

        for index, label in enumerate(dataset.get_labels()):
             if label in self.items_per_label.keys():
                self.items_per_label[label].append(index)
             else:
                self.items_per_label[label] = [index]

        #print('got labels for dataset')
        #print('stop')

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self):

        #only sample classes will be sampled, if none, all classes will be sampled
        if self.sample_classes is not None:
            sample_classes = set(self.sample_classes)
        else:
            sample_classes = self.items_per_label.keys()

        for _ in range(self.n_tasks):
            #print('iterator called')
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(sample_classes, self.n_way)
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data):
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
                - the bounding box of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """
       # print('first input data is')
        all_labels = list({x[1] for x in input_data})
        all_bbs = list({x[2] for x in input_data})
        # print(input_data)
        # print(input_data[0])
        #all_labels = input_data[0][1]

        #print(f'label ids are {all_labels[0:5]} ...')
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])

        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[2:])
        )
        
        max_pixel = torch.max(all_images)
        min_pixel = torch.min(all_images)
        mid = (max_pixel+min_pixel)/2
        #normalise
        all_images = (all_images-mid)/(max_pixel-min_pixel)

        true_class_ids = list(torch.unique(all_labels))
        all_labels = Tensor([true_class_ids.index(x) for x in all_labels])

        all_labels = all_labels.reshape(
             (self.n_way, self.n_shot + self.n_query))

        #convert class labels into ranking, so it can be indexed by one-hot encoding
        

        # true_class_ids = true_class_ids.reshape(
        #      (self.n_way, self.n_shot + self.n_query))

        #print(f'all labels shape is {all_labels.shape}')
        
        #print(f' all-images are {all_images[0:2]}...')
        # pylint: disable=not-callable
        # pylint: enable=not-callable
        #print(f' all-labels are {all_labels[0:2]}...')
        support_images = all_images[:, : self.n_shot].reshape((-1, *all_images.shape[2:]))
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))

        support_labels = all_labels[:,:self.n_shot].flatten()
        query_labels = all_labels[:,self.n_shot :].flatten()

        support_bbs = all_bbs[:,:self.n_shot]
        query_bbs = all_bbs[:,self.n_shot:]

        return (
            support_images,
            support_labels.long(),
            support_bbs,
            query_images,
            query_labels.long(),
            query_bbs,
            Tensor(true_class_ids).long()
        )

def convert_label(labels,im_width = 256,im_height = 256,margin=1):
    """
    Args: 
        label: takes label string list in format ['class, x_center, y_center, width, height','...','...']
        im_width: width of image in pixels 
        im_height: height of image in pixels
    Returns:
        class_label: str, concatenates labels in case of mutliple
        new_bb: [ymin,xmin,ymax,xmax]"""
    new_bbs = np.zeros((len(labels),4))
    class_labs = []
    class_concat = ''
    for i,lab in enumerate(labels):
        
        label_lst = lab.split()
        class_labs.append(label_lst[0])
        class_concat += (label_lst[0])
        #Yolo format: [x_center, y_center, width, height]
        bb = [float(i) for i in label_lst[1:]]
        x_centre = bb[0]*im_width
        y_centre = bb[1]*im_height
        width = bb[2]*im_width
        height = bb[3]*im_height
        #mew format: [ymin,xmin,ymax,xmax]
        new_bbs[i,:] = np.array([y_centre-height*margin/2,x_centre-width*margin/2,y_centre+height*margin/2,x_centre+width*margin/2])
    if len(labels)>1:
        new_bb = np.array([min(new_bbs[:,0]),min(new_bbs[:,1]),max(new_bbs[:,2]),max(new_bbs[:,3])])
    else:
        new_bb = new_bbs
    if class_concat=='':
        new_bb=np.array([0,0,0,0])
    return class_labs, new_bbs, np.squeeze(new_bb.astype(int)),class_concat

class industry_dataset(Dataset):
    def __init__(self, label_dir, img_dir, transforms=None, margin=1,im_transforms=None):
        """additional class to add bbs to FGVC-Aircraft
        Args:
          label_dir: file location of label files
          im_dir: file location of image files
          transform: transforms to be applied 
          file_name_to_bb_dict: gives file name to bb 
          transforms: Bool, whether to apply transforms to data, (limited by bounding boxes)
          
        Methods:
          __getitem__: Returns im, label, bb"""
        
        try:
            labels_df = pd.read_pickle(label_dir+'labels_file.pkl')
        except:
            image_filenames = [f[:-4] for f in listdir(img_dir) if f[-4:]=='.png']
            label_filenames = [f[:-4] for f in listdir(label_dir) if f[-4:]=='.txt']
            #get only images which have corresponding labels
            #file names are identical for images and labels
            file_names = pd.merge(pd.DataFrame(image_filenames),pd.DataFrame(label_filenames),how='inner').values.tolist()
            im_file_name_list = [f[0] +'.png' for f in file_names]
            self.label_file_name_list = [f[0]+'.txt' for f in file_names]
            #get list of class labels and bounding box labels
            self._labels = []
            self._clss_and_bb = []
            print('reading file labels...')
            for i in tqdm(range(len(image_filenames))):
                with open(label_dir+self.label_file_name_list[i]) as f:
                    label = f.readlines()
                _,_,new_bb,class_concat = convert_label(label,margin=margin)
                self._labels.append(class_concat)
                self._clss_and_bb.append((class_concat,new_bb))
            labels_df = pd.DataFrame((self._clss_and_bb),columns=('class_label','bb'))
            labels_df['file_name'] = im_file_name_list
            try:
                labels_df.to_pickle(label_dir+'labels_file.pkl')
            except:
                print('could not save labels to file')
            #TODO add code to save 
        labels_df = labels_df.fillna(-1)
        self._labels = list(labels_df['class_label'])
        self.labels_df = labels_df
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.im_transforms = im_transforms


    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx]['file_name'])
        im = tensor(cv2.imread(img_path))
        im = torch.moveaxis(im,2,0)
        label = self.labels_df.iloc[idx]['class_label']
        bb = self.labels_df.iloc[idx]['bb']

        if self.transforms:
            im, bb = transformsXY(torch.moveaxis(im,0,2), bb, self.transforms)
            im = torch.moveaxis(tensor(im),2,0)
        if self.im_transforms:
            im = self.im_transforms(im)
        im =transforms.Resize(size=(256,256))(im)
        return im,label,bb

    def get_labels(self):
        return self._labels
        

def get_extra_classes(all_classes,current_classes,n_additional_classes):
    """Function to return extra classes, not currently in dataset
    Args:
        all_classes: original classes in data_set: set(subset_ds.dataset._labels)
        current_classes: classes in subset dataset: set(subset_ds._labels) 
        n_additional_classes: number of additional classes to sample
    Returns:
        additional class labels"""
    available_classes = []
    print(f'len current classes is {len(current_classes)}')
    print(f'len all classes is {len(all_classes)}')
    for class_ in all_classes:
        if class_ in current_classes:
            continue
        else:
            available_classes.append(class_)
    print(f'available_classes are {available_classes}')
    new_classes = np.random.choice(available_classes,n_additional_classes,replace = False)
    return new_classes

def get_dict_of_labels(label_list):
    """
    Args:
        label_list: list of labels
    Returns:
        dictionary, with the indicies of each label"""
    label_index_dict = {}
    for index, label in enumerate(label_list):
        if label in label_index_dict:
            label_index_dict[label].append(index)
        else:
            label_index_dict[label] = [index]
    return label_index_dict

def get_new_class_indices(additional_classes,dataset_labels,n_shot):
    """Function to get indicies of new classes
    Args:
        additional_classes: list of new classes to be added
        dataset_labels: list of all labels in dataset. (dataset._labels)
        n_shot: number of samples from each class to be added to dataset
    Returns:
        list of new indicies of new classes"""
    label_dict = get_dict_of_labels(dataset_labels)
    new_indicies = []
    for label in additional_classes:
      try:
        new_indicies += list(np.random.choice(label_dict[label],n_shot,replace=False))
      except:
        #print(f'not enought samples in class {label} returning all {len(label_dict[label])} available samples ')
        new_indicies += label_dict[label]

    return new_indicies

def add_classes_to_dataset(subset_dataset,n_additional_classes,n_shot,original_labels=None,current_labels=None,new_classes=None):
    """Function to add samples of new classes
    Args
        subset_dataset: dataset of class Subset, must have .dataset and .indices attributes
        n_additional_classes: number of additional classes to add 
        n_shot: number of samples of each class to add 
        original_labels: list of orignal class labels, if dataset does not have ._labels attribute
        current_labels: list of current class labels of subset, if subset does not have ._labels attribute

    Returns
        subset dataset with original samples + additional classes"""
    if original_labels is None:
        original_labels = subset_dataset.dataset._labels
    if current_labels is None:
        current_labels = subset_dataset._labels
    if new_classes is None:
        new_classes = get_extra_classes(set(original_labels),set(current_labels),n_additional_classes)
    new_indices = get_new_class_indices(new_classes,original_labels,n_shot)
    new_indices += list(subset_dataset.indices)
    augmented_subset_dataset= Subset(subset_dataset.dataset,new_indices)
    augmented_subset_dataset._labels = np.array(original_labels)[new_indices]
    augmented_subset_dataset._new_classes =  np.unique(new_classes)
    augmented_subset_dataset._old_classes =  np.unique(current_labels)
    #base_classes = all classes in training set
    augmented_subset_dataset._base_classes = np.unique(augmented_subset_dataset._labels)

    return augmented_subset_dataset

def create_few_shot_subset(subset_dataset,n_way,n_shot,original_labels=None,current_labels=None,new_classes=None):
    """Function to add samples of new classes
    Args
        subset_dataset: dataset of class Subset, must have .dataset and .indices attributes
        n_way: number of classes
        n_shot: number of samples of each class to add 
        original_labels: list of orignal class labels, if dataset does not have ._labels attribute
        current_labels: list of current class labels of subset, if subset does not have ._labels attribute

    Returns
        subset dataset with original samples + additional classes"""
    if original_labels is None:
        original_labels = subset_dataset.dataset._labels
    if current_labels is None:
        current_labels = subset_dataset._labels
    if new_classes is None:
        new_classes = get_extra_classes(np.unique(current_labels), [],n_way)
    new_indices = get_new_class_indices(new_classes,original_labels,n_shot)
    augmented_subset_dataset= Subset(subset_dataset.dataset,new_indices)
    augmented_subset_dataset._labels = np.array(original_labels)[new_indices]
    augmented_subset_dataset._new_classes =  np.unique(new_classes)
    augmented_subset_dataset._old_classes =  np.unique(current_labels)
    #base_classes = all classes in training set
    augmented_subset_dataset._base_classes = np.unique(augmented_subset_dataset._labels)

    return augmented_subset_dataset

def get_sample_images(label_dict,dataset,n_shot,base_classes):
    """
    Args:
        label_dict: dictiorary of indicies of each class
        dataset: dataset from which to return images, must return (im, label)
        n_shot: number of sample from each class 
    Returns:
        sample images and labels"""
    #initialise empty arrays
    n_items = len(label_dict)*n_shot
    im_size = dataset.__getitem__(0)[0].size()
    sample_ims = torch.empty(n_items,*im_size)
    sample_labels = torch.empty(n_items)
    sample_bbs = torch.empty(n_items,4)
    #sample indices for each label
    for i,label in enumerate(label_dict):
        indices = np.random.choice(label_dict[label],n_shot,replace=False)
        #for each index, add to empty array
        for j, index in enumerate(indices):
            im, label, bb = dataset.__getitem__(index)
            sample_ims[i*n_shot+j]= im
            sample_labels[i*n_shot+j] = int(base_classes.tolist().index(label))
            sample_bbs[i*n_shot+j] = tensor(bb)

    return sample_ims, sample_labels, sample_bbs

def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def get_subset_classes(N_classes = 10,subset_classes = 5):
    """Returns subset of classes"""
    few_shot_classes = np.random.choice(N_classes,subset_classes,replace=False)
    base_classes = []
    for i in np.arange(N_classes):
        if i in few_shot_classes:
            continue
        else:
            base_classes.append(i)
    base_classes = np.array(base_classes)
    return base_classes, few_shot_classes

def subsetclasses(dataset,labels,base_classes):
    """Function to return subset of dataset, with base classes
    Args:
        dataset: input dataset 
        labels: list/array of dataset labels
        base_classes: ids of base classes to be preserved
    Returns:
        Subset of original dataset class """
    n_data = dataset.__len__()
    base_class_indicies = np.array([i for i in range(n_data) if labels[i] in base_classes])
    #from torch.utils.data import Subset
    subset_class = Subset(dataset,base_class_indicies)
    #add labels attribute 
    subset_class._labels = np.array(labels)[base_class_indicies]
    subset_class._base_classes = base_classes


    return subset_class

def custom_random_split(dataset,labels,lengths, generator=default_generator):
    """Modified from https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Subset
    Function to return random splits of dataset, with labels attributes (unlike pytorch function)
    Args:
        dataset: input dataset 
        labels: list of dataset labels
        lengths: must sum to original lenght of data set, must only be 2 lengths
    Returns:
        Subset of original dataset class """
    labels = np.array(labels)
    n_data = dataset.__len__()
    # Cannot verify that dataset is Sized
    assert sum(lengths) == n_data, "Sum of input lengths does not equal the length of the input dataset!"   
    
    indices = randperm(sum(lengths), generator=generator).tolist()

    split1 = Subset(dataset,indices[:lengths[0]])
    split1._labels = labels[indices[:lengths[0]]]
    split2 = Subset(dataset,indices[lengths[0]:])
    split2._labels = labels[indices[lengths[0]:]]
    return split1,split2


def subsetter(dataset,labels,base_classes,few_shot_class, n_shot, random_seed =None, data_frac =1):
    """Function to return subset of dataset, with all base classes randomly undersampled, and random sample of few_shot_class
    Args:
        dataset: input dataset 
        labels: list/array of dataset labels
        base_classes: ids of base classes to be preserved 
        few_shot_class: additional class to be added 
        n_shot: number of instances of few_shot class
    Returns:
        Subset of original dataset class"""
    n_data = dataset.__len__()
    labels = np.array(labels)
    label_dict = get_dict_of_labels(labels)
    min_class_freq = int(data_frac*min([len(i) for i in label_dict.values()]))
    base_class_indicies = np.zeros(len(base_classes)*min_class_freq)
    if random_seed:
        np.random.seed(random_seed)
    #randomly undersample each class to min class frequency
    for i, class_ in enumerate(base_classes):
        base_class_indicies[i*min_class_freq:(i+1)*min_class_freq] = np.random.choice(label_dict[class_],min_class_freq,replace=False)

    few_shot_indicies = [i for i in range(n_data) if labels[i]==few_shot_class]
    selected_few_shot_indicies = np.random.choice(few_shot_indicies,n_shot,replace=False)
    new_indicies = np.hstack((base_class_indicies.astype(int),selected_few_shot_indicies))
    new_dataset = Subset(dataset,new_indicies)
    new_dataset._labels = labels[new_indicies]
    #from torch.utils.data import Subset
    return new_dataset


def balanace_preds_list(ground_truths,preds):
    """Function to return class balanced list of ground truths and predictions
    Args:
        -ground_truths: list/array/tensor of ground truth classes
        -preds: list/array/tensor of predictions
    Returns:
        -balanced_ground_truths: np array of ground truths balanced by random oversampling
        -balanced_preds: np array of predictions balanced by random oversampling"""
    ground_truths = np.array(ground_truths)
    preds = np.array(preds)
    label_dict = get_dict_of_labels(ground_truths)
    max_class_freq = max([len(i) for i in label_dict.values()])
    indices = np.zeros(max_class_freq*len(label_dict)).astype(int)
    for i, key in enumerate(label_dict):
        indices[i*max_class_freq:(i+1)*max_class_freq] = np.random.choice(label_dict[key],max_class_freq,replace=True)
    balanced_ground_truths = ground_truths[indices]
    balanced_preds = preds[indices]

    return balanced_ground_truths, balanced_preds

def create_mask_tensor(bb, x):
    """Creates a mask for the bounding box of same shape as image
    x must be of shape (3,height,width)"""
    Y = torch.zeros_like(x)
    bb = bb.long()
    Y[:,bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def tensor_mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    print(Y.shape)
    print(Y[0].shape)
    cols, rows = torch.nonzero(Y[0],as_tuple=True) # returns a tuple of arrays, one for each dimension of arr, containing the indices of the non-zero elements in that dimension
    if len(cols)==0: 
        return torch.zeros(4, dtype=torch.float32)
    top_row = torch.min(rows)
    left_col = torch.min(cols)
    bottom_row = torch.max(rows)
    right_col = torch.max(cols)
    return tensor([left_col, top_row, right_col, bottom_row], dtype=torch.float32)


def augment_bbs(ims,bbs,augment_policy):
    """
    Args:
        -ims
        -bbs
        -augmentation_policy
    Returns:
        -augmented ims
        -augmented bbs"""
    Masks = torch.zeros_like(ims)
    for i,bb in enumerate(bbs):
        Masks[i] = create_mask_tensor(bb.long(),ims[i])
    ims_and_masks = torch.cat((ims,Masks))
    ims_and_masks = augment_policy(ims_and_masks)
    aug_ims = ims_and_masks[:len(ims)]
    aug_masks = ims_and_masks[len(ims):]
    aug_bbs = torch.zeros_like(bbs)
    for i,mask in enumerate(aug_masks):
        aug_bbs[i] = tensor_mask_to_bb(mask)
    return aug_ims, aug_bbs


#Reading an image
def read_image(path):
    #print(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def create_mask_tensor(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    if x.shape[0]==3:
        rows,cols = x.shape[1:3]
    else:
        rows,cols,*_ = x.shape
    Y = torch.zeros((rows, cols))
    #bb = bb.astype(int) #cast a pandas object to a specified dtype
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    if x.shape[0]==3:
        rows,cols = x.shape[1:3]
    else:
        rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    #bb = bb.astype(int) #cast a pandas object to a specified dtype
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y) # returns a tuple of arrays, one for each dimension of arr, containing the indices of the non-zero elements in that dimension
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

def bb_crop(im,bb):
    return im[bb[0]:bb[2],bb[1]:bb[3]]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(x, bb, transforms):
    #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    x = np.array(x)
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb, bb2=None):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    if bb2 is not None:
        plt.gca().add_patch(create_corner_rect(bb2,color='green'))

# def setup():
#     df_all = generate_train_df(anno_path)

#     #label encode target
#     class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
#     df_all['class'] = df_all['class'].apply(lambda x:  class_dict[x])


#     print(df_all)
#     print(df_all.shape)
#     print(df_all.sample(5))
    
#     #Populating Training DF with new paths and bounding boxes
#     new_paths = []
#     new_bbs = []
#     train_path_resized = Path('./road_signs/images_resized')
#     if not os.path.isdir( train_path_resized ) :
#         os.mkdir( train_path_resized )  # make sure the directory exists
#     for index, row in df_all.iterrows():
#         new_path,new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values),300)
#         new_paths.append(new_path)
#         new_bbs.append(new_bb)
#     df_all['new_path'] = new_paths
#     df_all['new_bb'] = new_bbs

#     im = cv2.imread(str(df_all.values[0][0]))
#     bb = create_bb_array(df_all.values[0])

#     print(f' bb is {bb}')
#     print(im.shape)

#     Y = create_mask(bb, im)
#     mask_to_bb(Y)

#     plt.imshow(im)

#     plt.show()

#     plt.imshow(Y, cmap='gray')

#     plt.show()

#     df_all.loc[0]

#     torch.save(df_all,'road_sign_all_data.pt')

def save_ims_as_arrays():
    df = torch.load('road_sign_all_data.pt')
    new_df = df[['class','new_bb','new_path']]
    new_df['image']=new_df['new_path'].apply(lambda x : cv2.imread(x))
    torch.save(new_df,'all_data_tensors.pt')

def tensors_to_rgb():
    df = torch.load('all_data_tensors.pt')

    df.rename(columns={'new_bb':'bb','new_path':'path'},inplace = True)

    def convert_to_rgb(im):
        return np.stack((im[:,:,2],im[:,:,1],im[:,:,0]),axis= -1)

    df['image']= df['image'].apply(convert_to_rgb)
    torch.save(df,'all_data_arrays.pt')

def train_val_test_split():
    df = torch.load('rgb_tensors.pt')
    df = df.sample(frac=1).reset_index(drop=True)
    
    n_data = len(df)
    train_split = 0.7
    train_index = int(n_data*train_split)
    val_split = 0.1
    val_index = train_index+int(val_split*n_data)
    test_split = 0.2

    print('saving training data')
    torch.save(df.iloc[:train_index],'road_signs/Tensor_datasets/training_data.pt')
    print('saving val data')
    torch.save(df.iloc[train_index:val_index],'road_signs/Tensor_datasets/val_data.pt')
    print('saving test data')
    torch.save(df.iloc[val_index:],'road_signs/Tensor_datasets/test_data.pt')

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]

class RoadDataset(Dataset):
    def __init__(self, x, bb, y, transforms=False):
        self.transforms = transforms
        self.x = x
        self.bb = bb.values
        self.y = y.values
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(x, self.bb[idx], self.transforms)
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb

    def get_labels(self):
        return self.y



class bb_dataset(Dataset):
    def __init__(self, dataset, file_list, file_name_to_bb_path='FGVCAircraft/fgvc-aircraft-2013b/data/images_box.txt',transforms=False):
        """additional class to add bbs to FGVC-Aircraft
        Args:
          dataset: input data set
          file_list: dataset._image_files tells the file name to index
          file_name_to_bb_dict: gives file name to bb 
          transforms: whether to do data augmentation
        Methods:
          __getitem__: Returns im, label, bb
        """
        self.__dict__ = dataset.__dict__
        self.dataset = dataset
        self.file_list = file_list
        self.transforms = transforms

        #read file_name_to_bb, store in dictionary, self.file_name_bb_dct
        with open(file_name_to_bb_path) as f:
          lines = f.readlines()
        
        self.file_name_bb_dct = {}
        for line in lines:
          bb_lst = line.split()
          self.file_name_bb_dct[int(bb_lst[0])] = [int(i) for i in bb_lst[1:]]

    def get_file_name(self,idx):
        return int(self.file_list[idx].split('/')[-1].split('.')[0])

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        im, label = self.dataset.__getitem__(idx)
        bb = self.file_name_bb_dct[self.get_file_name(idx)]
        #[ymin,xmin,ymax,xmax]
        bb = np.array([bb[1],bb[0],bb[3],bb[2]])
        if self.transforms:
            im, bb = transformsXY(torch.moveaxis(im,0,2), bb, self.transforms)
            im = torch.moveaxis(tensor(im),2,0)
        _, rows, cols = im.size()
        x_scale = 256/cols
        y_scale = 256/rows
        im =transforms.Resize(size=(256,256))(im)
        #[ymin,xmin,ymax,xmax]
        new_bb = np.array([int(bb[0]*y_scale),int(bb[1]*x_scale),int(bb[2]*y_scale),int(bb[3]*x_scale)])
        return im,label,new_bb

    def get_labels(self):
        return self._labels


def calc_IOU(bb1,bb2,im):
    m1 = create_mask(np.array(bb1),im)
    m2 = create_mask(np.array(bb2),im)
    intersection = np.logical_and(m1, m2)
    union = np.logical_or(m1, m2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# def crop_resize_from_bbs(ims,bbs,width=300,height=300):
#     """crops array of images based on bbs, outputs tensor """
#     assert len(ims.shape)==4, (f'expected array of multiple images, ims shape was {ims.shape}')

#     bbs = np.array(bbs)
#     ims = np.array(ims)
#     ims = np.moveaxis(ims,1,3)
#     # bbs = bbs.int()
#     # ims = torch.moveaxis(ims,1,3)

#     resized_ims = torch.empty((len(ims),width,height,3))
#     #resized_ims = torch.empty((len(ims),width,height,3)
#     for i, im in enumerate(ims):
#        # print(f'bbs are {bbs[i]}')
#         bb = ((abs(bbs[i])+bbs[i])/2).astype(int)
#         im = bb_crop(im,bb)
#        # print(im.shape)
#        # im = torch.moveaxis(im,3,1)
#        # print(im.size())
#         #im = TF.resize(im,(width,width)))
#         im = tensor(cv2.resize(im,(width,width)))
#         resized_ims[i] = im

#     return torch.moveaxis(resized_ims,3,1)

def bb_crop_resize(ims,bbs,width=256,height=256):
    """Args:
        ims: squence of images with shape [n_ims,n_channels,height,width] 
            or [n_channels,height,width] if 1 image
        bbs: sequence or single bounding box in format [ymnin,xmin,ymax,xmax]"""
    if len(ims.shape)==4:
        resized_ims = torch.empty((ims.shape[0],ims.shape[1],height,width))
        bbs = ((abs(bbs)+bbs)/2).int()
        for i, im in enumerate(ims):
            resized_ims[i] = TF.resize(im[:,bbs[i][0]:bbs[i][2],bbs[i][1]:bbs[i][3]],(height,width))
        return resized_ims
    elif len(ims.shape)==3:
        return TF.resize(ims[:,bbs[0]:bbs[2],bbs[1]:bbs[3]])
    else:
        raise Exception(f"expected images to have 3 or 4 dimensions, {len(ims.shape)} were found.")

def check_class_balance(input,class_name_list=None,plot=True, dataset_input = False):
    """checks class labels, plots bar chart,outputs dict of class distribution
    input: list of labels of data points, or dataset"""
    values = {}
    if dataset_input:
        for i in range(input.__len__()):
            _,label = input.__getitem__(i)
            if label in values.keys():
                values[label]+=1
            else:
                values[label]=1
        
    else:
        for i in input:
            if i in values.keys():
                values[i]+=1
            else:
                values[i]=1

    if plot:
        if class_name_list:
            class_names = [class_name_list[i] for i in values.keys()]
        else:
            class_names = values.keys()
        
        plt.bar(class_names,values.values())
        plt.show()

    return values
