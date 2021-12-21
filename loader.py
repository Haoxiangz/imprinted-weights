import torch
import os
import numpy as np
import pandas as pd
import math
import pickle
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, num_classes=50, num_train_sample=0, novel_only=False, aug=False,
                 loader=pil_loader):
        # here root is the folder name
        # img_folder = os.path.join(root, "images")
        # img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        # img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        # train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        if train:
            filename = './cifar-100-python/train'
            train_flag = True
        else:
            filename = './cifar-100-python/test'
            train_flag = False
        with open(filename,'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            data_size = len(dict[b'fine_labels'])
            # load images into the dataframe as PIL image
            # Because there will be transforms performed on them
            # idk a easier way to make this work
            dict['imgs'] = []
            for i in range(data_size):
                tmp_img = dict[b'data'][i].reshape(3,32,32)
                im = Image.fromarray(np.swapaxes(np.swapaxes(tmp_img, 0, 2),0,1), 'RGB')
                dict['imgs'].append(im)
            data = pd.concat([pd.DataFrame(dict[b'filenames']),pd.DataFrame(dict[b'fine_labels'])\
                ,pd.DataFrame(0,index=np.arange(data_size),columns=[2]),\
                pd.DataFrame(dict['imgs'])],axis=1)
            data.set_axis(['filenames','label','train_flag','data'],axis=1,inplace=True)

        # data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        # data = data[data['train_flag'] == train]
        # data['label'] = data['label'] - 1

        # split dataset
        data = data[data['label'] < num_classes]
        base_data = data[data['label'] < 50]
        novel_data = data[data['label'] >= 50]

        # sampling from novel classes
        if num_train_sample != 0:
            novel_data = novel_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[:num_train_sample])
            # if train:
            #     base_data = base_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[:num_train_sample])

        # whether only return data of novel classes
        if novel_only:
            data = novel_data
        else:
            data = pd.concat([base_data, novel_data])

        # repeat 5 times for data augmentation
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, data])
            data = tmp_data
        imgs = data.reset_index(drop=True)
        
        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        # self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[int(index)]
        # file_path = item['path']
        target = item['label']

        # img = self.loader(os.path.join(self.root, file_path))
        img = item['data']

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_balanced_sampler(self):
        img_labels = np.array(self.imgs['label'].tolist())
        class_sample_count = np.array([len(np.where(img_labels==t)[0]) for t in np.unique(img_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t-50] for t in img_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler