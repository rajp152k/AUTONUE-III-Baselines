import os
import time
import numpy as np
import torch , sys
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageOps
from argparse import ArgumentParser
from EntropyLoss import EmbeddingLoss
from iouEval import iouEval

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torch.nn.functional as F


from dataset_loader import *
import transform as transforms

import importlib
from collections import OrderedDict , namedtuple

from shutil import copyfile

class load_data():

        def __init__(self, args):

                ## First, a bit of setup
                dinf = namedtuple('dinf' , ['name' , 'n_labels' , 'func' , 'path', 'size'])
                if args.level==2:

                        self.metadata = [dinf('IDD', 17, IDD_Dataset , 'IDD_20k' , (1024,512)),
                                                dinf('CS' , 17 , CityscapesDataset , 'all_sources/cityscapes_sampled' , (1024,512)),
                                                dinf('Map', 17, Mapillary,'all_sources/mapillary_stratified',(1024,512)),
                                                dinf('gta', 17, Gta,'all_sources/gta_stratified',(1024,512)),
                                                dinf('bdds', 17, Bdds,'all_sources/bdds_stratified',(1024,512))
                                                ]
                else:
                        self.metadata = [dinf('IDD', 27, IDD_Dataset , 'IDD_20k' , (1024,512)),
                                                dinf('CS' , 27 , CityscapesDataset , 'all_sources/cityscapes_sampled' , (1024,512)) ,
                                                dinf('Map', 27, Mapillary,'all_sources/mapillary_stratified',(1024,512)),
                                                dinf('gta', 27, Gta,'all_sources/gta_stratified',(1024,512)),
                                                dinf('bdds', 27, Bdds,'all_sources/bdds_stratified',(1024,512))
                                                ]

                self.num_labels = {entry.name:entry.n_labels for entry in self.metadata if entry.name in args.datasets}

                self.d_func = {entry.name:entry.func for entry in self.metadata}
                basedir = args.basedir
                self.d_path = {entry.name:basedir+entry.path for entry in self.metadata}
                self.d_size = {entry.name:entry.size for entry in self.metadata}

        def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False, level=2,test = False):

                transform = self.Img_transform(name, self.d_size[name] , split)
                return self.d_func[name](self.d_path[name] , split, transform, file_path, num_images , mode, level, test)

        def Img_transform(self, name, size, split='train'):
                return transforms.Compose([transforms.ToTensor()])
                assert (isinstance(size, tuple) and len(size)==2)

                if name in ['CS' , 'IDD']:

                        if split=='train':
                                t = [
                                        transforms.Resize(size),
                                        transforms.RandomCrop((512,512)), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()]
                        else:
                                t = [transforms.Resize(size),
                                        transforms.ToTensor()]

                        return transforms.Compose(t)

                if split=='train':
                        t = [transforms.Resize(size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()]
                else:
                        t = [transforms.Resize(size),
                                transforms.ToTensor()]

                return transforms.Compose(t)

def test(args, get_dataset, model, enc=False):

        # if args.resume:
        #       #Must load weights, optimizer, epoch and best value. 
        #       if enc:
        #               filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        #       else:
        #               filenameCheckpoint = '/raid/cs18btech11021/semi_supervise/All_level3/checkpoint.pth.tar'

        #       assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        #       checkpoint = torch.load(filenameCheckpoint)
        #       start_epoch = checkpoint['epoch']
        #       model.load_state_dict(checkpoint['state_dict'])
        #       optimizer.load_state_dict(checkpoint['optimizer'])
        #       best_acc = checkpoint['best_acc']
        #       label_embedding = torch.load(le_file) if len(datasets) >1 else None
                # print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

        n_gpus = torch.cuda.device_count()
        print("\nWorking with {} GPUs".format(n_gpus))

        datasets = args.datasets
        NUM_LABELS = get_dataset.num_labels

        dataset_test = {dname: get_dataset(dname, 'test', args.num_samples,level = args.level, test = True) for dname in datasets}

        loader_test = {dname:DataLoader(dataset_test[dname], num_workers=args.num_workers, batch_size=args.batch_size, 
                                                        shuffle=False, drop_last=True) for dname in datasets}

        model.eval()
                                
        for d in datasets:
                time_taken = []    


                for itr, items in enumerate(loader_test[d]):

                        start_time = time.time()
                        images,filename = items
                        filename=  filename[0]
                        images = images.cuda()

                        with torch.set_grad_enabled(False):

                                # print(type(images))

                                seg_output = model(images, enc=False,finetune=args.finetune)
                                # loss = loss_criterion[d](seg_output[d], targets.squeeze(1))
                                pred = seg_output[d].argmax(1,True).data
                                #pred = Image.fromarray(np.array(pred.cpu(),dtype=np.uint8).squeeze())
                                pred = pred.cpu().data.numpy()
                                pred = Image.fromarray(pred.squeeze().astype(np.uint8))
                                writelabel(pred,filename,args.savedir)
                                #cv2.imwrite(pred,Path(args.savedir)/f'
                                print(f'done with {1+itr} images')

def writelabel(pred,imgname,saveroot):
    skip = len('/raid/datasets/SemanticSegmentation/domain_adaptation/IDD_20k/')
    imgname = f'{saveroot}/TEST/{imgname[skip:]}'
    os.makedirs(Path(imgname).parent,exist_ok=True)
    pred.save(Path(imgname))




def main(args, get_dataset):
        # savedir = f'../save_{args.model}/{args.savedir}'
        savedir = args.savedir

        # if os.path.exists(savedir + '/model_best.pth') and not args.resume and not args.finetune:
        #       print("Save directory already exists ... ")
        #       sys.exit(0)

        if not os.path.exists(savedir):
                os.makedirs(savedir)

        if not args.resume:
                with open(savedir + '/opts.txt', "w") as myfile:
                        myfile.write(str(args))

        #Load Model
        assert os.path.exists(args.model + ".py"), f"Error: model definition for {args.model} not found"

        model_file = importlib.import_module(args.model)
        if args.bnsync:
                model_file.BatchNorm = batchnormsync.BatchNormSync
        else:
                model_file.BatchNorm = torch.nn.BatchNorm2d


        NUM_LABELS = get_dataset.num_labels

        model = model_file.Net(NUM_LABELS , args.em_dim , args.resnet)          
        # copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

        # if args.cuda:
        #       model = torch.nn.DataParallel(model).cuda()
        
        if args.state:
                
                def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
                        own_state = model.state_dict()
                        state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
                        for name, param in state_dict.items():
                                
                                if name.startswith(('seg' , 'up' , 'en_map' , 'en_up')):
                                        continue
                                elif name not in own_state:
                                        print("Not loading {}".format(name))
                                        continue
                                own_state[name].copy_(param)

                        print("Loaded pretrained model ... ")
                        return model

                
                state_dict = torch.load(args.state)
                model = load_my_state_dict(model, state_dict)
                model.cuda()


        # train_start = time.time()

        # model = train(args, get_dataset, model, False)

        test(args, get_dataset, model, False)
        # print("========== TRAINING FINISHED ===========")
        # print(f"Took {(time.time()-train_start)/60} minutes")


def parse_args():

        parser = ArgumentParser()
        parser.add_argument('--model')
        parser.add_argument('--debug' , action='store_true')
        parser.add_argument('--basedir', required=True)
        parser.add_argument('--bnsync' , action='store_true')
        parser.add_argument('--lr' , required=True, type=float)
        parser.add_argument('--random-rotate' , type=int, default=0)
        parser.add_argument('--random-scale' , type=int, default=0)
        parser.add_argument('--num-epochs', type=int, default=150)
        parser.add_argument('--batch-size', type=int, default=6)
        parser.add_argument('--savedir', required=True)
        parser.add_argument('--datasets' , nargs='+', required=True)
        parser.add_argument('--em-dim', type=int, default=100)
        parser.add_argument('--K' , type=float , default=1e4)
        parser.add_argument('--theta' , type=float , default=0)
        parser.add_argument('--num-samples' , type=int) ## Number of samples from each dataset. If empty, consider full dataset.
        parser.add_argument('--update-embeddings' , type=int , default=0)
        parser.add_argument('--pt-em')
        parser.add_argument('--alpha' , type=int, required=True) ## Cross dataset loss term coeff.
        parser.add_argument('--beta' , type=int , required=True) ## Within dataset loss term coeff. 
        parser.add_argument('--resnet' , required=True)
        parser.add_argument('--pAcc' , action='store_true')
        parser.add_argument('--level', type = int, default=2)

        ### Optional ######
        parser.add_argument('--finetune' , action='store_true')
        parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
        parser.add_argument('--state')
        parser.add_argument('--port', type=int, default=8097)
        parser.add_argument('--height', type=int, default=512)
        parser.add_argument('--num-workers', type=int, default=2)
        parser.add_argument('--steps-loss', type=int, default=50)
        parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
        parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
        parser.add_argument('--iouVal', action='store_true', default=True)  
        parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

        args = parser.parse_args()

        return args


if __name__ == '__main__':

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

        try:
                args = parse_args()
                get_dataset = load_data(args)
                main(args, get_dataset)
        except KeyboardInterrupt:
                sys.exit(0)
