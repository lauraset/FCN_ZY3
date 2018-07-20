#FCN
#STEP 1 load data including image and groudtruth
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tifffile as tif
import matplotlib.pyplot as plt
import numpy as np
import math
import  random
input_data_path='/home/wubantu/yinxcao/ZYDATA/'
input_data_size=1
fold_size=3
win_stride=112
patch_size=224
channels=3
num_class=7

class Dataset(object):
    batch_offset = 0
    epochs_completed = 0
    def __init__(self,imgdata,gtdata,num_examples):
        self._imgdata=imgdata
        self._gtdata=gtdata
        self._num_examples=num_examples
        self._height=imgdata.shape[1]
        self._width=imgdata.shape[2]
    @property
    def imgdata(self):
        return self._imgdata
    @property
    def gtdata(self):
        return self._gtdata
    @property
    def num_examples(self):
        return  self._num_examples
    @property
    def height(self):
        return  self._height
    @property
    def width(self):
        return  self._width
    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self._imgdata.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self._imgdata.shape[0])
            np.random.shuffle(perm)
            self._imgdata = self._imgdata[perm]
            self._gtdata = self._gtdata[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self._imgdata[start:end], self._gtdata[start:end]
# read whole image (imagezy and groundtruth)
def read_tiff(file_path,file_size):
    for i in range(0,file_size):
        imgdata=tif.imread([file_path+'IMG_'+str(i+1)+'.tif'])
        gtdata=tif.imread([file_path+'GT_'+str(i+1)+'.tif'])

    dataset=Dataset(imgdata,gtdata,file_size)
    print("image size is :")
    print(dataset.imgdata.shape)
    return dataset
#make train/validataion/test data shuffle
#cross-validation 3-fold
def gen_sample(dataset,win_stride,patch_size):
    height=dataset.height
    width=dataset.width
    rows=math.floor((height-patch_size)/win_stride+1)
    cols=math.floor((width-patch_size)/win_stride+1)
    totoalsample=rows*cols
    print('totoalsample is %d'%totoalsample)
    image_list=list(range(1,totoalsample+1))
    random.shuffle(image_list)
    per_size=math.floor(totoalsample/(fold_size+1))
    image_directory={'image_list':image_list,'rows':rows,'cols':cols}
    return image_directory
#generate train validation test samples
def extract_sample(dataset,image_list,rows,cols):
    num_image=len(image_list)
    images=np.zeros((num_image,patch_size,patch_size,channels),dtype=np.float32)
    labels = np.zeros((num_image, patch_size, patch_size, 1), dtype=np.int32)
    for i in range(0,len(image_list)):
        id=image_list[i]
        id_i=math.floor(id/cols)+1-1
        id_j=id-(id_i-1)*cols-1
        imgdata=dataset.imgdata[0:3, id_i:(id_i+patch_size),id_j:(id_j+patch_size)].copy()
        imgdata=np.transpose(imgdata,(1,2,0))
        #imagedata need regularization
        #[0-65535]->[0,1]
        gtdata=dataset.gtdata[id_i:(id_i+patch_size),id_j:(id_j+patch_size)].copy()
        #labels need one-hot coding
        gtdata=np.reshape(gtdata,(patch_size,patch_size,1))
        images[i,:,:,:]=imgdata
        labels[i,:,:,:]=gtdata-1
    channel_mean=images.mean(axis=(0,1,2))
    channel_std=images.std(axis=(0,1,2))
    images=(images-channel_mean)/channel_std
    res=Dataset(images,labels,num_image)
    return res

# dataset=read_tiff(input_data_path,input_data_size)
# imagedire=gen_sample(dataset,win_stride,patch_size)
# list=imagedire['image_list']
# rows=imagedire['rows']
# cols=imagedire['cols']
# per_size=math.floor(len(list)/(fold_size+1))
# trainset=extract_sample(dataset,list[0:per_size*2],rows,cols)
# valset=extract_sample(dataset,list[per_size*2:per_size*3],rows,cols)
# testset=extract_sample(dataset,list[per_size*3:],rows,cols)

