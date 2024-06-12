# Command line instructions needed
##### Clone the repo #####
# !git clone 'https://github.com/WongKinYiu/yolov9.git'


##### needed because there is an error in the repo #####
# !sed -i 's/opt.min_items/min_items/' yolov9/val.py
# !sed -i 's/opt.min_items/min_items/' yolov9/val_dual.py

##### Install the requirements #####
# !pip install -r yolov9/requirements.txt -q

import sys
import os
import yaml
import random
import torch

sys.path.append('../yolov9')

# if you are using BranchyYOLO
from train import main as train
from val import main as test

# # if you are using ablated YOLO
# from train_dual import main as train
# from val_dual import main as test

# Useful paths
save_dir = 'dir_train/' # path to the folder where the result of the training will be saved
local_path = '' # path to folder with all images
img_path = local_path + 'synthetic_images/images/'
labels_path = local_path + 'synthetic_images/labels/'
real_images_path = local_path + 'real_images/images/'
real_labels_path = local_path + 'real_images/labels/'

###### classes ####

class Opt:
    def __init__(self, *args, **kwargs):
        self.project = save_dir+'runs/'
        self.name = 'trainingModel'
        self.weights = ''
        self.data = 'coco.yaml' # train/val/test data saved into yaml
        self.exist_ok = False
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.nosave = False
        self.imgsz = 640

        for key, value in kwargs.items():
            setattr(self, key, value)


class TrainOpt(Opt):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = 'train'
        self.cfg = 'BranchyYOLO.yaml'   # model topology
        self.epochs = 300
        self.batch_size = 20
        self.evolve = False
        self.resume = False
        self.single_cls = False
        self.noval = False           # validation after each epoch
        self.workers = 8
        self.freeze = [0]
        self.noplots = False
        self.seed = 0
        self.optimizer = 'Adam'     
        self.cos_lr = False
        self.flat_cos_lr = False
        self.fixed_lr = False
        self.sync_bn = False
        self.cache = 'ram'          # disk creates the npy for images, ram then the model goes out of cuda memory
        self.close_mosaic = 20      # number of last epochs without using moasic
        self.rect = False         
        self.quad = False
        self.image_weights = False
        self.min_items = 0
        self.label_smoothing = 0.0
        self.patience = 50
        self.multi_scale = True
        self.save_period = -1      # used only if nosave is False
        self.hyp = 'hyp.yaml'

        for key, value in kwargs.items():
            setattr(self, key, value)


class TestOpt(Opt):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = 'test'
        self.conf_thres = 0.001
        self.save_hybrid = False
        self.task = self.name
        self.min_items = 0
        del self.nosave

        for key, value in kwargs.items():
            setattr(self, key, value)


####### Functions to define the datasets #######

def defineDatasetSynthetic(): 
    # dataset with only synthetic images
    # remove old cache if there is one
    try:
        os.remove('train.cache')
        os.remove('val.cache')
        os.remove(labels_path+'../labels.cache')
    except OSError:
        pass
    print('Deleted old cache correctly')

    # read how many images are in the folder and shuffle them
    imgs = [os.path.join(img_path, name) for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]
    random.shuffle(imgs)
    img_n = len(imgs)

    trainleng = int(0.7 * img_n)
    valLength = int(0.15 * img_n)

    # files used for training/validating/testing
    with open('train.txt', 'w') as f:
        f.write('\n'.join([str(imgs[i]) for i in range(0, trainleng)]))
    with open('val.txt', 'w') as f:
        f.write('\n'.join([str(imgs[i]) for i in range(trainleng, trainleng+valLength)]))
    with open('test.txt', 'w') as f:
        f.write('\n'.join([str(imgs[i]) for i in range(trainleng+valLength, img_n)]))

    # update coco file
    data = {}
    data['path'] = local_path
    data['train']= 'train.txt'
    data['val']= 'val.txt'
    data['test'] = 'test.txt'
    data['nc'] = 12
    data['names'] = {
        0: 'blue_army',
        1: 'red_army',
        2: 'yellow_army',
        3: 'purple_army',
        4: 'black_army',
        5: 'green_army',
        6: 'blue_flag',
        7: 'red_flag',
        8: 'yellow_flag',
        9: 'purple_flag',
        10: 'black_flag',
        11: 'green_flag',
    }

    with open('coco.yaml', 'w') as f:
        yaml.dump(data, f)
    print('Modified correctly')


def defineDatasetSyntheticReal():
    # dataset with synthetic images and real images
    # remove old cache
    try:
        os.remove('train.cache')
        os.remove('val.cache')
        os.remove(labels_path+'../labels.cache')
    except OSError:
        pass
    print('Deleted old cache correctly')

    # read how many images are in the folder and shuffle them
    synthetic_imgs = [os.path.join(img_path, name) for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]
    real_imgs = [os.path.join(real_images_path, name) for name in os.listdir(real_images_path) if os.path.isfile(os.path.join(real_images_path, name))]
    random.shuffle(synthetic_imgs)
    random.shuffle(real_imgs)

    img_synthetic_n = len(synthetic_imgs)
    img_real_n = len(real_imgs)
    trainleng = int(0.7 * img_synthetic_n)
    valLeng = int(0.15 * img_synthetic_n)
    trainRealLen = int(0.7 * img_real_n)

    # file used for training and validation
    with open('train2.txt', 'w') as f:
        f.write('\n'.join([str(synthetic_imgs[i]) for i in range(0, trainleng)]))
        f.write('\n')
        f.write('\n'.join([str(real_imgs[i]) for i in range(0, trainRealLen)]))
    with open('val2.txt', 'w') as f:
        f.write('\n'.join([str(synthetic_imgs[i]) for i in range(trainleng, trainleng + valLeng)]))

    # test2.txt contains only synthetic images, while test3.txt contains only real images
    with open('test2.txt', 'w') as f:
        f.write('\n'.join([str(synthetic_imgs[i]) for i in range(trainleng + valLeng, img_synthetic_n)]))
    with open('test3.txt', 'w') as f:
        f.write('\n'.join([str(real_imgs[i]) for i in range(trainRealLen, img_real_n)]))

    # update coco file
    data = {}
    data['path'] = local_path
    data['train']= 'train2.txt'
    data['val']= 'val2.txt'
    data['test'] = 'test2.txt'
    data['nc'] = 12
    data['names'] = {
        0: 'blue_army',
        1: 'red_army',
        2: 'yellow_army',
        3: 'purple_army',
        4: 'black_army',
        5: 'green_army',
        6: 'blue_flag',
        7: 'red_flag',
        8: 'yellow_flag',
        9: 'purple_flag',
        10: 'black_flag',
        11: 'green_flag',
    }

    with open('coco.yaml', 'w') as f:
        yaml.dump(data, f)
        print('Modified correctly')


def main():
########### FIRST PART #################

    # Train onto synthetic images
    train_opt = TrainOpt()

    defineDatasetSynthetic()

    try:
        train(train_opt)
    except Exception as error:
        torch.cuda.empty_cache()
        print("An error occurred:", type(error).__name__, "-", error)

    # Test onto synthetic images
    test_opt = TestOpt(weights=Opt().project+'train/weights/best.pt')
    test(test_opt)

    # update coco to test onto real images
    with open('coco.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    data['test'] = 'real_images/images'
    with open('coco.yaml', 'w') as f:
        yaml.dump(data, f)
    print('Modified correctly')

    # Test onto real images
    test_opt = TestOpt(weights=Opt().project+'train/weights/best.pt')
    test(test_opt)


########### SECOND PART #################
    # If both are run sequentially this training creates train2 folder in dir_train/runs
    # and the weights are saved in train2/weights otherwise change the path in the test_opt
    # Training onto real and synthetic images
    train_opt = TrainOpt()

    defineDatasetSyntheticReal()

    try:
        train(train_opt)
    except Exception as error:
        torch.cuda.empty_cache()
        print("An error occurred:", type(error).__name__, "-", error)

    # Test onto synthetic images
    test_opt = TestOpt(weights=Opt().project+'train2/weights/best.pt')
    test(test_opt)

    # update coco to test onto real images
    with open('coco.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    data['test'] = 'test3.txt'
    with open('coco.yaml', 'w') as f:
        yaml.dump(data, f)
        
    # Test onto real images
    test_opt = TestOpt(weights=Opt().project+'train2/weights/best.pt')
    test(test_opt)

if __name__=='__main__':
    main()



