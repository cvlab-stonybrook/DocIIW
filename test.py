'''
Code for inferring the WBNet results
'''
import os
import torch
import cv2
import numpy as np
import scipy.misc as m
from models.unetnc import UnetGenerator
from utils import *
from loss import *


testpath='./testimgs/real/'

model_path= './checkpoints/unet_173_0.0943109058694208_0.08592824839441238_wbkunet_train12_l1chroma_rot_scratch_best_model.pkl'
out='./results/{}/{}'.format(model_path.split('/')[-1],testpath.split('/')[-1])
print (out)

model=UnetGenerator(input_nc=3, output_nc=3, num_downs=7)
state = convert_state_dict(torch.load(model_path)['model_state'])
model.load_state_dict(state)
model.cuda(0)
model.eval()

def tight_crop(img,alb):
    msk=((img[:,:,0]!=0)&(img[:,:,1]!=0)&(img[:,:,2]!=0)).astype(np.uint8)
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

def infer(img_path, alb_path, mask=None, pad=False):
    print("Read Input Image from : {}".format(img_path))
    imgorg = m.imread(img_path,mode='RGB')
    if pad:
        imgorg = np.pad(imgorg, ((80, 80), (80, 80), (0, 0)), 'constant')
    print (np.max(imgorg))
    print (np.min(imgorg))
    alb=None
    if alb_path is not None:
        alb = m.imread(alb_path,mode='RGB')
        imgorg, alb = tight_crop(imgorg,alb)
        alb = alb.astype(float)/255.
    # plt.imshow(imgorg)
    # plt.show()
    img=imgorg.astype(float)/255.0
    img = cv2.resize(img, (256,256)) 
    if mask is not None:
        mask = cv2.resize(mask, (256,256)) 
        img=img*mask
    img = img[:, :, ::-1] # RGB -> BGR
    img = img.astype(np.float32)
    if img.shape[2] == 4:
        img=img[:,:,:3]
    imgt = img.transpose(2, 0, 1) # NHWC -> NCHW
    imgt = torch.from_numpy(imgt).unsqueeze(0).float().cuda(0)


    with torch.no_grad():
        outputs = model(imgt)
        wbk_outputs = outputs
        wbk_outputs=wbk_outputs.detach().cpu().numpy()

    # NCHW->NWHC
    wbk_outputs=np.transpose(wbk_outputs,[0,2,3,1])[0]
    wbk_outputs=wbk_outputs[:,:,::-1]  #BGR -> RGB

    wbk_outputs=cv2.resize(wbk_outputs, (imgorg.shape[1], imgorg.shape[0]))
    wb_outputs=cv2.normalize(wbk_outputs,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*imgorg.astype(float)/255.0
    wb_outputs=cv2.normalize(wb_outputs,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    wb_outputs_bgr=wb_outputs[:,:,::-1]  #RGB -> BGR
    print (os.path.join(out,img_path.split('/')[-1]))
    cv2.imwrite(os.path.join(out,img_path.split('/')[-2]+'-'+img_path.split('/')[-1]),wb_outputs_bgr*255)


if not os.path.exists(out):
    os.makedirs(out)

for fn in os.listdir(testpath):
    if '.png' in fn or '.jpg' in fn: 
        # if 'another' not in fn:
        #     continue 
        img_path=os.path.join(testpath,fn)
        mask_path=img_path.replace('images','masks')
        # print (mask_path)
        msk=cv2.imread(mask_path,0)
        msk[msk>0]=1.0
        msk=np.expand_dims(msk, -1)
        msk=np.concatenate([msk,msk,msk],axis=-1)
        res=infer(img_path, alb_path=None, mask=msk)
