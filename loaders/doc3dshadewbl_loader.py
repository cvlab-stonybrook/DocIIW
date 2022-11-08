# loader for light regression
import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import hdf5storage as h5
import random

from tqdm import tqdm
from torch.utils import data

class Doc3dshadewblLoader(data.Dataset):
    """
    Data loader for the  doc3d-shade dataset.
    """
    def __init__(self, root, split='train', is_transform=True, img_size=256, aug=False):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.augmentations=aug
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        if aug:
            self.texroot='/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/'
            self.bg_texpaths=open(os.path.join(self.texroot,'augtexnames.txt'),'r').read().split('\n')

        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #2/4-0_-12.5-iliad_Page_630-Nov0001-L1_4-T_15500-I_9306.png
        im_path = pjoin(self.root, 'img',  im_name)  
        img_foldr,fname=im_name.split('/')
        imname_s=im_name.split('-')
        alb_name=('-').join(imname_s[:-3])

        wbl_path = pjoin(self.root, 'wbl' , im_name)
        alb_path = pjoin(self.root, 'alb' , alb_name + '.png')
        sn_path = pjoin(self.root, 'norm' , alb_name + '.exr')

        
        img = m.imread(im_path,mode='RGB')
        wbl = m.imread(wbl_path,mode='RGB')
        alb = m.imread(alb_path,mode='RGB')
        snorm = cv2.imread(sn_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if snorm is None:
            print (sn_path)

        if self.augmentations:
            img,wbl,alb,snorm=self.aug(img,wbl,alb,snorm)

        if self.is_transform:
            d = self.transform(img,wbl,alb,snorm)
        return d

    def rotateImage(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center,angle,1.0)
        rotated_image = cv2.warpAffine(image, M, (w,h))
        return rotated_image
    
    def aug(self,img, wbl, alb, snorm):
        # horz flip, vertical flip, rotation
        #flipVert
        prob=np.random.uniform()
        # print (prob)

        # if prob < 0.3:
        #     img = cv2.flip(img, 0)
        #     wbl = cv2.flip(wbl, 0)
        #     alb = cv2.flip(alb, 0)
        #     snorm = cv2.flip(snorm, 0)

        # #flipHorz
        # elif prob >= 0.2 and prob <0.5:
        #     img = cv2.flip(img, 1)
        #     wbl = cv2.flip(wbl, 1)
        #     alb = cv2.flip(alb, 1)
        #     snorm = cv2.flip(snorm, 1)
        
        # #flipBoth
        # elif prob >= 0.5 and prob <0.7:
        #     img = cv2.flip(img, -1)
        #     wbl = cv2.flip(wbl, -1)
        #     alb = cv2.flip(alb, -1)
        #     snorm = cv2.flip(snorm, -1)

        # rotate
        if prob >= 0.5:
            ang=random.uniform(-40,40)
            img=self.rotateImage(img, ang)
            wbl=self.rotateImage(wbl, ang)
            alb=self.rotateImage(alb, ang)
            snorm=self.rotateImage(snorm, ang)

        return img,wbl,alb,snorm



    def tight_crop(self, alb, img, wbl, sn):
        msk1=((sn[:,:,0]!=0)&(sn[:,:,1]!=0)&(sn[:,:,2]!=0)).astype(np.uint8)
        msk2=((img[:,:,0]!=0)&(img[:,:,1]!=0)&(img[:,:,2]!=0)).astype(np.uint8)
        msk=np.bitwise_or(msk1, msk2)
        # print (np.max(msk1))
        size=msk.shape
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        img = img[miny : maxy + 1, minx : maxx + 1, :]
        wbl = wbl[miny : maxy + 1, minx : maxx + 1, :]
        sn = sn[miny : maxy + 1, minx : maxx + 1, :]
        msk = msk[miny : maxy + 1, minx : maxx + 1]
        
        s = 25
        img = np.pad(img, ((s, s), (s, s), (0, 0)), 'constant')
        wbl = np.pad(wbl, ((s, s), (s, s), (0, 0)), 'constant')
        sn = np.pad(sn, ((s, s), (s, s), (0, 0)), 'constant')
        msk = np.pad(msk, ((s, s), (s, s)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        img = img[cy1 : -cy2, cx1 : -cx2, :]
        wbl = wbl[cy1 : -cy2, cx1 : -cx2, :]
        sn = sn[cy1 : -cy2, cx1 : -cx2, :]
        msk = msk[cy1 : -cy2, cx1 : -cx2]

        return img, wbl, msk, sn

    def preproc_img(self, img):
        # img = img / 255.0
        img = cv2.resize(img, self.img_size) 
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float32)
        if img.shape[2] == 4:
            img=img[:,:,:3]
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        return img

    def color_jitter(self, im, wb,  brightness=0, contrast=0, saturation=0, hue=0):
        f = random.uniform(1, 1 + contrast)
        im = np.clip(im * f, 0., 1.)
        wb = np.clip(wb * f, 0., 1.)
        f = random.uniform(0.0, brightness)
        im = np.clip(im + f, 0., 1.).astype(np.float32)
        wb = np.clip(wb + f, 0., 1.).astype(np.float32)
        # hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # f = random.uniform(-hue, hue)*360.
        # hsv[:,:,0] = np.clip(hsv[:,:,0] + f, 0., 360.)
        # f = random.uniform(-saturation, saturation)
        # hsv[:,:,1] = np.clip(hsv[:,:,1] + f, 0., 1.)
        # im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return im, wb

    def replace_bg(self,msk,im,bg):
        # replace bg
        bg=bg.astype(float)/255.0
        msk = np.expand_dims(msk, axis=2)
        [fh, fw, _] = im.shape
        chance=random.random()
        if chance > 0.3:
            bg = cv2.resize(bg, (200, 200))
            bg = np.tile(bg, (3, 3, 1))
            bg = bg[: fh, : fw, :]
        elif chance < 0.3 and chance> 0.2:
            c = np.array([random.random(), random.random(), random.random()])
            bg = np.ones((fh, fw, 3)) * c
        else:
            bg=np.zeros((fh, fw, 3))
            msk=np.ones((fh, fw, 3))
        im = bg * (1 - msk) + im * msk
        return im


    def transform(self, img, wbl, alb, snorm):
        img, wbl, msk, snorm=self.tight_crop(alb, img, wbl, snorm)
        wbl=wbl.astype(float)/255.0
        img=img.astype(float)/255.0
        # img, wbl= self.color_jitter(img, wbl, brightness=0.2, contrast=0.2)
        # if self.augmentations:
        #     tex_id=random.randint(1,5639)
        #     tex=cv2.imread(os.path.join(self.texroot,self.bg_texpaths[tex_id])).astype(np.uint8)
        #     bg=cv2.resize(tex,self.img_size)
        #     imgb=self.replace_bg(msk,img,bg)
        # else: 
        imgb=img.copy()
        if 'val' in self.split:
            imgb=img.copy()

        wbk= np.divide(wbl,img,out=np.zeros_like(wbl), where=img!=0)

        # normalize wbk
        # mx,my,mz,nx,ny,nz=64.95033333333333, 149.17569882352942, 212.8709126361655, 0.0, 0.7494979012345677, 0.0
        # wbk[:,:,0]=(wbk[:,:,0]-nx)/(mx-nx)
        # wbk[:,:,1]=(wbk[:,:,1]-ny)/(my-ny)
        # wbk[:,:,2]=(wbk[:,:,2]-nz)/(mz-nz)
        mnormx,mnormy,mnormz,nnormx,nnormy,nnormz=0.002576476, 0.90569293, 0.8339295, -0.99999994, -0.91040754, -0.91640145
        snorm[:,:,0]=(snorm[:,:,0]-nnormx)/(mnormx-nnormx)
        snorm[:,:,1]=(snorm[:,:,1]-nnormy)/(mnormy-nnormy)
        snorm[:,:,2]=(snorm[:,:,2]-nnormz)/(mnormz-nnormz)

        # print (np.max(wbk))
        # print (np.min(wbk))
        img = self.preproc_img(imgb)
        wbl = self.preproc_img(wbl)
        wbk = self.preproc_img(wbk)        
        msk = cv2.resize(msk, self.img_size) 
        msk = np.expand_dims(msk, 0).astype(float)
        snorm = cv2.resize(snorm, self.img_size)
        snorm = snorm.transpose(2, 0, 1)
        # snorm = snorm*np.concatenate([msk,msk,msk], axis=0)
        # print (np.max(img))
        # print (np.min(img))
        # print (np.max(wbl))
        # print (np.min(wbl))
        # wbk= np.divide(wbl,img,out=np.zeros_like(wbl), where=img!=0)
        # print (np.max(msk))
        # print (np.min(msk))


        img = torch.from_numpy(img).float()
        wbl = torch.from_numpy(wbl).float()
        wbk = torch.from_numpy(wbk).float()
        msk = torch.from_numpy(msk).float()
        snorm = torch.from_numpy(snorm).float()
        d={'img':img, 'wbl':wbl, 'wbk':wbk, 'msk':msk, 'snorm':snorm}
        return d


 
# Leave code for debugging purposes
if __name__ == '__main__':
    local_path = '/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/Doc3DShade/'
    bs = 4
    dst = Doc3dshadewblLoader(root=local_path, split='train', is_transform=True, aug=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs = data['img']
        wbls = data['wbl']
        wbks = data['wbk']
        msks = data['msk']
        # print (msks.shape)

        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0,2,3,1])[:,:,:,::-1]
        wbls = wbls.numpy()
        wbls = np.transpose(wbls, [0,2,3,1])[:,:,:,::-1]
        wbks = wbks.numpy()
        wbks = np.transpose(wbks, [0,2,3,1])[:,:,:,::-1]
        msks = msks.numpy()
        msks = np.transpose(msks, [0,2,3,1])[:,:,:,::-1]

        # mx,my,mz,nx,ny,nz=64.95033333333333, 149.17569882352942, 212.8709126361655, 0.0, 0.7494979012345677, 0.0
        # wbks[:,:,:,0]=(wbks[:,:,:,0]*(mx-nx))+nx
        # wbks[:,:,:,1]=(wbks[:,:,:,1]*(my-ny))+ny
        # wbks[:,:,:,2]=(wbks[:,:,:,2]*(mz-nz))+nz
        # return wbks

        f, axarr = plt.subplots(bs, 4)
        
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(wbls[j])
            axarr[j][2].imshow(msks[j][:,:,0])
            axarr[j][3].imshow(imgs[j]*wbks[j])
            print (np.max(imgs[j]*wbks[j]))
            print (np.min(imgs[j]*wbks[j]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()
