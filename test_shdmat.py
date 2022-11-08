'''
Code for inferring the SMTNet results
'''
import os
import argparse
import torch
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import scipy.misc as m

from models.uneted import UNetMatsm255,UNetMat 
from utils import *
from loss import *

testpath='./results/unet_173_0.0943109058694208_0.08592824839441238_wbkunet_train12_l1chroma_rot_scratch_best_model.pkl/real/'
model_path='./checkpoints/matunet_62_0.000184823025324165_0.015025632565304863_matshdunetsm255_train12_0.1-0.2l1chromal1penwb0.1shdpen_rot_0.5l1chromal1penwbgradrot_best_model.pkl'
# model_path='./checkpoints/matunet_70_0.0006177115177557076_0.3189237128553252_matshdunetsm255nsf_trainr_0.5l1chromal1pen[wbtex]advshdpen_rot_0.5l1chromal1penwb0.01texadvshd_best_model.pkl'

out='./results/{}'.format(model_path.split('/')[-1][:-4])
print (out)
if not os.path.exists(out):
    os.makedirs(out)

#setup model
model=UNetMatsm255(input_ch=3, output_ch=3)
# model=UNetMatsm255nsf(input_ch=3, output_ch=3)
state = convert_state_dict(torch.load(model_path)['model_state'])
model.load_state_dict(state)
model.cuda(0)
model.eval()

def tight_crop(img,alb):
    msk=((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(np.uint8)
    size=msk.shape
    [y, x] = (msk).nonzero()
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)

    img = img[miny : maxy + 1, minx : maxx + 1, :]
    alb = alb[miny : maxy + 1, minx : maxx + 1, :]
    
    s = 25
    img = np.pad(img, ((s, s), (s, s), (0, 0)), 'constant')
    alb = np.pad(alb, ((s, s), (s, s), (0, 0)), 'constant')
    cx1 = 5
    cx2 = 5
    cy1 = 5
    cy2 = 5

    img = img[cy1 : -cy2, cx1 : -cx2, :]
    alb = alb[cy1 : -cy2, cx1 : -cx2, :]

    return img, alb

def infer(img_path, alb_path, msk_path):
    print("Read Input Image from : {}".format(img_path))
    imgorg = m.imread(img_path,mode='RGB')
    alb=None
    if alb_path is not None:
        alb = m.imread(alb_path,mode='RGB')
        imgorg, alb = tight_crop(imgorg,alb)
        alb = alb.astype(float)/255.
        
    if msk_path is not None:
        msk = m.imread(msk_path,mode='RGB')
        imgorg, msk = tight_crop(imgorg,msk)
        msk = msk.astype(float)/255.
    # plt.imshow(imgorg)
    # plt.show()
    img=imgorg.astype(float)/255.0
    img = cv2.resize(img, (256,256)) 
    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(np.float32)
    if img.shape[2] == 4:
        img=img[:,:,:3]
    imgt = img.transpose(2, 0, 1) # NHWC -> NCHW
    imgt = torch.from_numpy(imgt).unsqueeze(0).float().cuda(0)

    with torch.no_grad():
        outputs = model(imgt)
        pred_mat=outputs['mat']
        pred_shd=outputs['shd']
        mat=pred_mat.detach().cpu().numpy()
        shd=pred_shd.detach().cpu().numpy()

    # NCHW->NWHC
    mat=np.transpose(mat,[0,2,3,1])[0]
    mat=mat[:,:,::-1]  #BGR -> RGB
    shd=np.transpose(shd,[0,2,3,1])[0]
    shd=shd[:,:,::-1]  #BGR -> RGB

    mat=cv2.resize(mat, (imgorg.shape[1], imgorg.shape[0]))
    shd=cv2.resize(shd, (imgorg.shape[1], imgorg.shape[0]))
    shdn=cv2.normalize(shd,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    matn=cv2.normalize(mat,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    msk=np.expand_dims(((shd[:,:,0]!=0)&(shd[:,:,1]!=0)&(shd[:,:,2]!=0)).astype(float),-1)

    if alb is not None:
        fig, axrr =plt.subplots(1,6)
        plt.tight_layout()
        axrr[0].imshow(imgorg.astype(float)/255)
        msk2=np.expand_dims(((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(float),-1)
        msk=msk+msk2
        msk[msk>0]=1.0
        axrr[1].imshow(alb)
        axrr[2].imshow(matn*msk)
        axrr[3].imshow(shdn)
        axrr[4].imshow(alb*matn)
        axrr[5].imshow(alb*matn*shdn)

    else:
        shdmat= shd*mat
        nonshd=np.clip(np.divide(imgorg,shdn,out=np.zeros_like(shdn), where=shdn!=0),0,255)
        nonshdmat=np.clip(np.divide(imgorg,shdmat,out=np.zeros_like(shdmat), where=shdmat!=0),0,255)
        # nonshdn=cv2.normalize(nonshd,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # plt.show()
    print (img_path.split('/')[-1])
    cv2.imwrite(os.path.join(out, img_path.split('/')[-1][:-4]+'-ns.png'), nonshd[:,:,::-1])
    # cv2.imwrite(os.path.join(out, img_path.split('/')[-1][:-4]+'-nsn.png'), nonshdn[:,:,::-1])
    cv2.imwrite(os.path.join(out, img_path.split('/')[-1][:-4]+'.png'), nonshdmat[:,:,::-1])
    cv2.imwrite(os.path.join(out, img_path.split('/')[-1][:-4]+'-shd.png'), shdn*255)


for fn in os.listdir(testpath):
    if '.png' in fn or '.jpg' in fn:
        img_path=os.path.join(testpath,fn)
        res=infer(img_path, alb_path=None,msk_path=None)

