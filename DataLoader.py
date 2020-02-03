
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
train_df['modes'] = 'train'

sub = pd.read_csv('sample_submission.csv')


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


class Dataset(data.Dataset):
    def __init__(self, root_path, df, transform=None, t_mode = 'train'):
        self.root_path = root_path
        self.records = df
        self.transform = transform
        self.t_mode = t_mode
        
    def __len__(self):
        return len(self.records)
    
    @staticmethod
    def load_image(path):
        pass
    
    def _get_img_path(self, index):
        pass
    
    def __getitem__(self, index):
        
        #load
        
        image_name = self.records.iloc[index].ImageId
        
        image = plt.imread(self.root_path + image_name)
        
        image = image / 255

        if self.records.iloc[index].modes == 'train':

          encoded_masks = train_df.loc[train_df['ImageId'] == image_name, 'EncodedPixels']
          masks = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)

          for idx, label in enumerate(encoded_masks.values):
            if label is not np.nan:
              mask = rle_decode(label)
              masks[:, :, idx] = mask
       
        else:

          encoded_masks = sub.loc[test_df['ImageId'] == image_name, 'EncodedPixels']
          masks = np.zeros((350, 525, 4), dtype=np.float32)

        
        #augmentation
        if self.transform is not None:
            image, masks = self.transform( (image, masks ) )

        image = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))
        masks = torch.from_numpy(np.transpose(masks, (2, 0, 1)).astype('float32'))
        
        return image, masks

