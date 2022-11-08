'''
Code to train WBNet
'''

import os
import argparse
import torch
from tqdm import tqdm

import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from tensorboardX import SummaryWriter

from models.unetnc import UnetGenerator
from loaders.doc3dshadewbl_loader import Doc3dshadewblLoader 
from loss import *


def train(args):
    logdir='./checkpoints/'
    
    arch='unet'
    root='/media/hilab/sagniksSSD/Sagnik/FoldedDocumentDataset/Doc3DShade/'
    experiment_name='wbkunet_train12_l1l1wbchroma_rot_l1l1wb-54' #model_data_loss_augmentation_trainstart
    writer = SummaryWriter(comment=experiment_name)

    #get dataloader
    l=Doc3dshadewblLoader(root=root, aug=True)
    lv=Doc3dshadewblLoader(root=root, split='val')
    trainloader=data.DataLoader(l, batch_size=args.batch, num_workers=5, shuffle=True)
    valloader=data.DataLoader(lv, batch_size=args.batch, num_workers=5)

    #get model
    model=UnetGenerator(input_nc=3, output_nc=3, num_downs=7)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    #optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4,amsgrad=True)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    epoch_start=1
    #look for checkpoints
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'], strict=False)
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 
    #loss
    # MSE=nn.MSELoss()
    L1=nn.L1Loss()
    smL1=nn.SmoothL1Loss()

    global_step=1
    #forward     
    avg_loss=0.0
    # avg_trloss=0.0
    best_val_loss=9999.0
    for epoch in range(epoch_start,args.epochs):
        train_loss=0.0
        train_chroma=0.0
        avg_loss=0.0
        model.train()
        for i, d in enumerate(trainloader):
            images=Variable(d['img'].cuda().float())
            wbs=Variable(d['wbl'].cuda().float())
            wbks=Variable(d['wbk'].cuda().float())
            msks=Variable(d['msk'].cuda().float())
            optimizer.zero_grad()
            preds=model(images)
            l1loss=L1(preds,wbks)
            chromaloss, pred_wbs=chromaticity_loss(preds,wbs,images,msks)
            l1loss_wb=L1(pred_wbs,wbs)
            loss=l1loss+l1loss_wb+chromaloss
            loss.backward()
            optimizer.step()

            # track losses
            avg_loss+=float(l1loss)
            train_loss+=float(l1loss)
            train_chroma+=float(chromaloss)

            if (i+1) % 100 == 0:
                avg_loss=avg_loss/100
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch,args.epochs,i, len(trainloader), avg_loss))
                avg_loss=0.0
            if (i+1) % 10 == 0:
                idxs=torch.LongTensor(6).random_(0, images.shape[0])
                grid_inp = torchvision.utils.make_grid(images[idxs],normalize=True, scale_each=True)
                grid_wbs_pred = torchvision.utils.make_grid(images[idxs]*preds[idxs],normalize=True, scale_each=True)
                grid_wbs_gt = torchvision.utils.make_grid(wbs[idxs],normalize=True, scale_each=True)
                grid_wbks_gt = torchvision.utils.make_grid(wbks[idxs],normalize=True, scale_each=True)
                grid_wbks_pred = torchvision.utils.make_grid(preds[idxs],normalize=True, scale_each=True)
                writer.add_image('inputs/train', grid_inp, global_step)
                writer.add_image('wb_pred/train', grid_wbs_pred, global_step)
                writer.add_image('wb_gt/train', grid_wbs_gt, global_step)
                writer.add_image('wbk_pred/train', grid_wbks_pred, global_step)
                writer.add_image('wbk_gt/train', grid_wbks_gt, global_step)
                writer.add_scalar('Loss/train', float(l1loss), global_step)
                writer.add_scalar('CLoss/train', float(chromaloss), global_step)
            global_step+=1
            # break
        # break
        train_loss=train_loss/len(trainloader)
        train_chroma=train_chroma/len(trainloader)
        print("Training WBK Loss:'{}'".format(train_loss))
        print("Training WB Loss:'{}'".format(train_chroma))

        #validation
        model.eval()
        # val_rot=0.0
        val_chroma=0.0
        val_loss=0.0
        for i, d in tqdm(enumerate(valloader)) :
            with torch.no_grad():
                images_val=Variable(d['img'].cuda().float())
                wbs_val=Variable(d['wbl'].cuda().float())
                wbks_val=Variable(d['wbk'].cuda().float())
                msks=Variable(d['msk'].cuda().float())
                preds_val=model(images_val)
                l1loss=L1(preds_val,wbks_val)
                chromaloss, pred_wbs_val=chromaticity_loss(preds_val,wbs_val,images_val,msks)
                val_loss+=float(l1loss)
                val_chroma+=float(chromaloss)
        val_loss=val_loss/len(valloader)
        val_chroma=val_chroma/len(valloader)
    
        idxs=torch.LongTensor(6).random_(0, images_val.shape[0])
        grid_inp = torchvision.utils.make_grid(images_val[idxs],normalize=True, scale_each=True)
        grid_wbs_pred = torchvision.utils.make_grid(images_val[idxs]*preds_val[idxs],normalize=True, scale_each=True)
        grid_wbs_gt = torchvision.utils.make_grid(wbs_val[idxs],normalize=True, scale_each=True)
        writer.add_image('inputs/val', grid_inp, global_step)
        writer.add_image('wb_pred/val', grid_wbs_pred, global_step)
        writer.add_image('wb_gt/val', grid_wbs_gt, global_step)
        writer.add_scalar('Loss/val', float(val_loss), global_step)
        writer.add_scalar('CLoss/val', float(val_chroma), global_step)

        print("Validation WBK Loss:'{}'".format(val_loss))
        print("Validation WB Loss:'{}'".format(val_chroma))
        sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            state = {'epoch': epoch,'model_state': model.state_dict()}
            torch.save(state, logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(arch,epoch,val_loss,train_loss,experiment_name))
        if (epoch % 5)==0:
            state = {'epoch': epoch,'model_state': model.state_dict()}
            torch.save(state, logdir+"{}_{}_{}_{}_{}_model.pkl".format(arch,epoch,val_loss,train_loss,experiment_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--imgsize', nargs='?', type=int, default=256, help='image size')
    parser.add_argument('--epochs', nargs='?', type=int, default=100, help='num of epochs')
    parser.add_argument('--batch', nargs='?', type=int, default=50, help='batch size')
    parser.add_argument('--resume', nargs='?', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0001, help='Learning rate')
    args = parser.parse_args()

    #print model
    train(args)
