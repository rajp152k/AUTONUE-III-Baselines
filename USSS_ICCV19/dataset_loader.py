import numpy as np
import os
import random

from PIL import Image
import torch

from torch.utils.data import Dataset
import glob


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class SegmentationDataset(Dataset):
    
    def __init__(self, root, subset,
                img_path, label_path, pattern, img_suffix, label_suffix,  file_path=False, transform=None, num_images=None,level=2,test = False):

        self.test = test

        # print(img_path)
        if "IDD_20k" in root:
            if self.test == False:
                self.image_file = f'{root}/semi-images-1005.txt'
                self.image_file_ul = f'{root}/rest-images-13022.txt'
                self.label_file=''
                if level == 2:
                    self.label_file = f'{root}/semi-labels-level2-1005.txt'
                else:
                    self.label_file = f'{root}/semi-labels-level3-1005.txt'

                self.image_paths=[]
                self.label_paths=[]
                for line in open(self.image_file,'r'):
                    self.image_paths.append(line.strip())
                if self.mode == 'unlabeled':   
                    for line in open(self.image_file_ul,'r'):
                        self.image_paths.append(line.strip())
                for line in open(self.label_file,'r'):
                    self.label_paths.append(line.strip())
                
            else:
                print("loda")
                self.image_file = f'{root}/rest-images-13022.txt'
                self.image_file_ul = f'{root}/rest-images-13022.txt'
                self.image_paths=[]
                self.label_paths=[]
                for line in open(self.image_file,'r'):
                    self.image_paths.append(line.strip())
                for line in open(self.image_file_ul,'r'):
                    self.label_paths.append(line.strip())
                leng = len(self.image_paths)
                print(self.image_paths[0])
                self.label_paths = self.label_paths[:leng]
                print(self.label_paths[0])
                print(len(self.image_paths),len(self.label_paths))
            
        else:
            self.images_root = f'{root}/{img_path}/{subset}'
            self.labels_root = f'{root}/{label_path}/{subset}'
            # print(self.images_root)
            self.image_paths = glob.glob(f'{self.images_root}/{pattern}')

            self.label_paths = [ img.replace(self.images_root, self.labels_root).replace(img_suffix, label_suffix) for img in self.image_paths  ]
        # if "IDD_20k" in root:
        #     self.image_paths = self.image_paths[:4000]
        #     self.label_paths = self.label_paths[:4000]
        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]

        self.num_classes = 16 if level == 2 else 26

        self.file_path = file_path
        self.transform = transform
        self.relabel = Relabel(255, self.num_classes) if transform != None else None


    def __getitem__(self, index):

        filename = self.image_paths[index]
        # if self.mode == 'labeled' :
        filenameGt = self.label_paths[index]


        with Image.open(filename) as f:
            image = f.convert('RGB')

        if self.mode == 'labeled':
            with Image.open(filenameGt) as f:
                label = f.convert('P')
        else:
            label = image


        # print(image.size, label.size)
        if self.transform !=None:
            image, label = self.transform(image, label)

        if self.test==True:
            return image,filename


        if self.relabel != None and self.mode == 'labeled':
            label = self.relabel(label)

        if self.phase_test == True:
            return image
        if self.mode == 'unlabeled':
            return image
        else:
            return image, label


    def __len__(self):
        return len(self.image_paths)

class CityscapesDataset(SegmentationDataset):

    label_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    color_map = np.array([
        [128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        [  0,  0,142],
        [  0,  0, 70],
        [  0, 60,100],
        [  0, 80,100],
        [  0,  0,230],
        [119, 11, 32]
    ], dtype=np.uint8)


    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled',level=2,test = False):
        self.d_idx = 'CS'
        self.mode = mode
        self.phase_test = test
        if level == 2:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level2', pattern='/*',
                    img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelIds_level2.png', transform=transform, file_path=file_path, num_images=num_images,test = test)
        else:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level3', pattern='/*',
                    img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelIds_level3.png', transform=transform, file_path=file_path, num_images=num_images, test = test)

class IDD_Dataset(SegmentationDataset):

    label_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']

    color_map   = np.array([
        [128, 64, 128], #road
        [ 81,  0, 81], #drivable fallback
        [244, 35, 232], #sidewalk
        [152, 251, 152], #nondrivable fallback
        [220, 20, 60], #pedestrian
        [255, 0, 0],  #rider
        [0, 0, 230], #motorcycle
        [119, 11, 32], #bicycle
        [255, 204, 54], #autorickshaw
        [0, 0, 142], #car
        [0, 0, 70], #truck
        [0, 60, 100], #bus
        [136, 143, 153], #vehicle fallback
        [220, 190, 40], #curb
        [102, 102, 156], #wall
        [190, 153, 153], #fence
        [180, 165, 180], #guard rail
        [174, 64, 67], #billboard
        [220, 220, 0], #traffic sign
        [250, 170, 30], #traffic light
        [153, 153, 153], #pole
        [169, 187, 214], #obs-str-bar-fallback
        [70, 70, 70], #building
        [150, 120, 90], #bridge
        [107, 142, 35], #vegetation
        [70, 130, 180] #sky
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled',level=2,test = False):
        self.d_idx = 'IDD'
        self.mode = mode
        self.phase_test = test
        if level == 2:
            super().__init__(root, subset,  
                    img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                    img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel2Ids.png', transform=transform, file_path=file_path, num_images=num_images,level=level,test = test)
        else:
            super().__init__(root, subset,  
                    img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                    img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel3Ids.png', transform=transform, file_path=file_path, num_images=num_images,level=level,test = test)

class Mapillary(SegmentationDataset):

    # make label and color code

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled',level=2):
        self.d_idx = 'Map'
        self.mode = mode
        if level == 2:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level2', pattern='/*',
                    img_suffix = '.jpg' , label_suffix='_level2Ids.png', transform=transform, file_path=file_path, num_images=num_images)
        else:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level3', pattern='/*',
                    img_suffix = '.jpg' , label_suffix='_level3Ids.png', transform=transform, file_path=file_path, num_images=num_images)

class Gta(SegmentationDataset):

    # make label and color code

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled',level=2):
        self.d_idx = 'gta'
        self.mode = mode
        if level == 2:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level2', pattern='/*',
                    img_suffix = '.png' , label_suffix='_level2.png', transform=transform, file_path=file_path, num_images=num_images)
        else:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level3', pattern='/*',
                    img_suffix = '.png' , label_suffix='_level3.png', transform=transform, file_path=file_path, num_images=num_images)

class Bdds(SegmentationDataset):

    # make label and color code

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled',level=2):
        self.d_idx = 'bdds'
        self.mode = mode
        if level == 2:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level2', pattern='/*',
                    img_suffix = '.jpg' , label_suffix='_train_id__trainLevel2Ids.png', transform=transform, file_path=file_path, num_images=num_images)
        else:
            super().__init__(root, subset,  
                    img_path = 'images', label_path='labels/level3', pattern='/*',
                    img_suffix = '.jpg' , label_suffix='_train_id__trainLevel3Ids.png', transform=transform, file_path=file_path, num_images=num_images)

def colorize(img, color, fallback_color=[0,0,0]): 
    img = np.array(img)
    W,H = img.shape
    view = np.tile(np.array(fallback_color, dtype = np.uint8), (W,H, 1) )
    for i, c in enumerate(color):
        indices = (img == i)
        view[indices] = c
    return view

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    

    def show_data(ds):
        print(len(ds))
        i = random.randrange(len(ds))
        img, gt = ds[i]
        color_gt = colorize(gt, ds.color_map)
        print(img.size,color_gt.shape)
        plt.imshow(img)
        plt.imshow(color_gt, alpha=0.25)
        plt.show()


    # cs = CityscapesDataset('/ssd_scratch/cvit/girish.varma/dataset/cityscapes')
    # show_data(cs)

    # an = ANUEDataset('/ssd_scratch/cvit/girish.varma/dataset/anue')
    # show_data(an)

    # bd = BDDataset('/ssd_scratch/cvit/girish.varma/dataset/bdd100k')
    # show_data(bd)

    # mv = MVDataset('/ssd_scratch/cvit/girish.varma/dataset/mvd')
    # show_data(mv)
