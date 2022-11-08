'''
Code to train SMTNet with Adversarial Loss
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
import numpy as np
import kornia

from models.uneted import UNetMatsm255, SpatialAttn, UNetMatsm255nsf
from models.patchgan import NLayerDiscriminator
from loaders.doc3dshadewblref_loader import Doc3dshadewblrefLoader
from loss import *
from utils import *

def train(args):
    logdir='./checkpoints/'
    arch='matunet'
    dc_arch='patchgan'
    root='/media/hilab/sagniksSSD/Sagnik/FoldedDocumentDataset/Doc3DShade/'
    experiment_name='matshdunetsm255nsf_trainr_0.5l1chromal1pen[wbtex]advshdpen_rot_0.5l1chromal1penwb0.01texadvshd' #model_data_loss_augmentation_trainstart
    if args.logtb:
        writer = SummaryWriter(comment=experiment_name)

    #get dataloader
    l=Doc3dshadewblrefLoader(root=root, img_size=256, aug=True)
    lv=Doc3dshadewblrefLoader(root=root, split='val', img_size=256)
    trainloader=data.DataLoader(l, batch_size=args.batch, num_workers=5, shuffle=True)
    valloader=data.DataLoader(lv, batch_size=args.batch, num_workers=5)

    #get model
    model= UNetMatsm255nsf(input_ch=3, output_ch=3)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    dc_model= NLayerDiscriminator(3, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, use_sigmoid=False)
    dc_model = torch.nn.DataParallel(dc_model, device_ids=range(torch.cuda.device_count()))
    dc_model.cuda()

    #optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4,amsgrad=True)
    dc_optimizer= torch.optim.Adam(dc_model.parameters(),lr=args.l_rate_dc)

    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    dc_sched=torch.optim.lr_scheduler.ReduceLROnPlateau(dc_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    epoch_start=1
    #look for checkpoints
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'], strict=False)
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']+1
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    # Look for Discrminator checkpoint 
    if args.dc_resume is not None:                                         
        if os.path.isfile(args.dc_resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.dc_resume))
            checkpoint = torch.load(args.dc_resume)
            dc_model.load_state_dict(checkpoint['model_state'], strict=False)
            print("Loaded checkpoint '{}' (epoch {})".format(args.dc_resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.dc_resume)) 
    #loss
    # MSE=nn.MSELoss()
    L1=nn.L1Loss()
    L1sum=nn.L1Loss(reduction='sum')

    #activation
    htan=nn.Hardtanh(0,1)

    global_step=1
    #forward     
    avg_loss=0.0
    # avg_trloss=0.0
    best_val_loss=9999.0
    eps=1e-2
    best_val_wd=9999.0
    # print epoch_start
    one = torch.tensor(1.0).cuda()
    mone = torch.tensor(-1.0).cuda()
    dc_iter=5

    for epoch in range(epoch_start,args.epochs):
        train_loss=0.0
        train_chroma=0.0
        train_wd=0.0
        avg_loss=0.0
        model.train()
        dc_model.train()
        for i, d in enumerate(trainloader):
            wbs=Variable(d['wbl'].cuda().float())
            msks=Variable(d['msk'].cuda().float())
            albs=Variable(d['alb'].cuda().float())
            set_requires_grad(dc_model, False)
            set_requires_grad(model, True)
            optimizer.zero_grad()
            dc_optimizer.zero_grad()
            pred=model(wbs)
            pred_mat=pred['mat']
            pred_shd=pred['shd']
            #calculate alb with material
            pred_alb=torch.mul(pred_mat,albs)
            #calculate shd with material
            pred_matshd=torch.clamp(torch.mul(pred_mat, pred_shd),0,255.)/255.0
            #predicted wb image
            pred_wbs=torch.mul(pred_alb,pred_shd)
            #predicted texture
            pred_tex=torch.clamp(torch_masked_divide(wbs, pred_matshd)/255.0, 0, 1.0)
            #predicted wbtex image
            pred_wbtex=torch.mul(pred_matshd,pred_tex)
            #estimated shading
            shd_gt=Variable(torch.clamp(torch_masked_divide(wbs, pred_alb), 0, 255.0))
            shd_gt = kornia.rgb_to_grayscale(shd_gt).expand_as(pred_shd)
            chromaloss=chromaticity_mat_mse(pred_alb, pred_mat, wbs, msks)
            constloss=L1(pred_shd,shd_gt)
            # gloss, gt_grad=grad_loss(pred_shd, shd_gt, reduction='mean', invalid_msk=None)
            wbloss=L1(pred_wbs, wbs) #TODO: scale invariant loss
            wbtexloss=L1(pred_wbtex, wbs) 
            texloss=L1(pred_tex, albs)
            shdpen= shading_l1_penalty(pred_shd)
            if epoch > 3:
                loss=chromaloss+(0.5*constloss)+wbtexloss+(0.05*shdpen)#+eloss
            elif epoch>1 and epoch <= 3:
                loss=chromaloss+(0.5*constloss)+wbtexloss#+(0.1*shdpen)
            else: 
                loss=chromaloss
            
            if epoch > 1:
                #backward adv shd
                # predicted shading should fake the discriminator
                pred_fake = dc_model(pred_shd)
                # print (pred_fake.shape)
                loss_shd_gan = torch.mean(pred_fake)
                loss_shd_gan.backward(mone, retain_graph=True)
            
            loss.backward()
            optimizer.step()

            # track losses
            avg_loss+=float(loss)
            train_loss+=float(loss)
            train_chroma+=float(chromaloss)

            if epoch > 1:
                set_requires_grad(dc_model, True)
                set_requires_grad(model, False)

                for dc_i in range(dc_iter):
                    # Update discriminator
                    dc_optimizer.zero_grad()
                    # Real
                    pred_real = dc_model(shd_gt.detach())
                    # print pred_real.shape
                    loss_D_real = torch.mean(pred_real)
                    # print loss_D_real
                    loss_D_real.backward(mone, retain_graph=True)

                    #backward D
                    pred_fake = dc_model(pred_shd.detach())
                    # print pred_fake.shape
                    loss_D_fake = torch.mean(pred_fake)
                    loss_D_fake.backward(one, retain_graph=True)

                    # calculate gradient penalty
                    gradient_penalty = calc_gradient_penalty(dc_model, shd_gt.detach(), pred_shd.detach())
                    gradient_penalty.backward()
                    # Combined loss
                    loss_D = loss_D_fake - loss_D_real + gradient_penalty
                    W_D = loss_D_fake - loss_D_real
                    dc_optimizer.step()
                train_wd+=float(W_D)
            else:
                loss_D=loss_shd_gan=W_D=0.0

            if (i+1) % 100 == 0:
                avg_loss=avg_loss/100
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch,args.epochs,i, len(trainloader), avg_loss))
                avg_loss=0.0
            if (i+1) % 10 == 0 and args.logtb:
                # preds_d=denormalize_wbks(preds.detach())
                # wbks_d=denormalize_wbks(wbks.detach())
                idxs=torch.LongTensor(6).random_(0, wbs.shape[0])
                grid_alb_pred = torchvision.utils.make_grid(pred_alb[idxs],normalize=True, scale_each=True)
                grid_wbs_gt = torchvision.utils.make_grid(wbs[idxs],normalize=True, scale_each=True)
                grid_wbs_pred = torchvision.utils.make_grid(pred_wbtex[idxs],normalize=True, scale_each=True)
                grid_shd_gt = torchvision.utils.make_grid(shd_gt[idxs],normalize=True, scale_each=True)
                grid_shd_pred = torchvision.utils.make_grid(pred_shd[idxs],normalize=True, scale_each=True)
                grid_tex_pred = torchvision.utils.make_grid(pred_tex[idxs],normalize=True, scale_each=True)
                grid_tex_gt = torchvision.utils.make_grid(albs[idxs],normalize=True, scale_each=True)

                writer.add_image('wb_inp/train', grid_wbs_gt, global_step)
                writer.add_image('wb_pred/train', grid_wbs_pred, global_step)
                writer.add_image('alb_pred/train', grid_alb_pred, global_step)
                writer.add_image('shd_gt/train', grid_shd_gt, global_step)
                writer.add_image('shd_pred/train', grid_shd_pred, global_step)
                writer.add_image('tex_gt/train', grid_tex_gt, global_step)
                writer.add_image('tex_pred/train', grid_tex_pred, global_step)
                writer.add_scalar('Loss/train', float(loss), global_step)
                writer.add_scalar('CLoss/train', float(chromaloss), global_step)
                writer.add_scalar('CnLoss/train', float(constloss), global_step)
                writer.add_scalar('WbLoss/train', float(wbloss), global_step)
                writer.add_scalar('TexLoss/train', float(texloss), global_step)
                writer.add_scalar('disc_loss/train', float(loss_D), global_step)
                writer.add_scalar('gen_loss/train', float(loss_shd_gan), global_step)
                writer.add_scalar('em_dist/train', float(W_D), global_step)
            global_step+=1
            # break
        # break
        train_chroma=train_chroma/len(trainloader)
        train_loss=train_loss/len(trainloader)
        train_wd=train_wd/len(trainloader)
        print("Training Chroma Loss:'{}'".format(train_chroma))

        #validation
        model.eval()
        dc_model.eval()
        val_chroma=0.0
        val_const=0.0
        val_loss=0.0
        val_pos=0.0
        val_wb=0.0
        val_tex = 0.0
        val_wd = 0.0
        val_gan = 0.0

        for i, d in tqdm(enumerate(valloader)):
            with torch.no_grad():
                wbs_val=Variable(d['wbl'].cuda().float())
                msks=Variable(d['msk'].cuda().float())
                albs_val=Variable(d['alb'].cuda().float())
                pred_val=model(wbs_val)
                pred_mat_val=pred_val['mat']
                pred_shd_val=pred_val['shd']
                pred_alb_val=torch.mul(pred_mat_val,albs_val)
                # shd_val_gt=Variable(torch.clamp(torch.div(wbs_val,pred_alb_val+eps),0,1.0))
                pred_wbs_val=torch.mul(pred_alb_val,pred_shd_val)
                pred_matshd_val=torch.clamp(torch.mul(pred_mat_val, pred_shd_val),0,255.)/255.0
                # pred_matshd_cpu=pred_matshd_val.detach().cpu().numpy()
                # pred_alb_cpu=pred_alb_val.detach().cpu().numpy()
                # wbs_cpu=wbs_val.detach().cpu().numpy()
                # pred_tex_val=torch.from_numpy(np.divide(wbs_cpu,pred_matshd_cpu,out=np.zeros_like(pred_matshd_cpu), where=pred_matshd_cpu!=0)).cuda().float()
                pred_tex_val=torch.clamp(torch_masked_divide(wbs_val, pred_matshd_val)/255.0, 0, 1.0)
                # pred_wbtex_val=torch.mul(pred_matshd_val,pred_tex_val)
                # shd_val_gt= Variable(torch.from_numpy(np.divide(wbs_cpu,pred_alb_cpu,out=np.zeros_like(pred_alb_cpu), where=pred_alb_cpu!=0)).cuda().float())
                shd_val_gt=Variable(torch.clamp(torch_masked_divide(wbs_val, pred_alb_val), 0, 255.0))
                shd_val_gt = kornia.rgb_to_grayscale(shd_val_gt).expand_as(pred_shd_val)
                chromaloss=chromaticity_mat_mse(pred_alb_val, pred_mat_val, wbs_val, msks)
                constloss=L1(pred_shd_val, shd_val_gt)
                wbloss=L1(pred_wbs_val, wbs_val)
                # wbtexloss=L1(pred_wbtex_val, wbs_val)
                texloss=L1(pred_tex_val, albs_val)
                val_chroma+=float(chromaloss)
                val_const+=float(constloss)
                val_wb+=float(wbloss)
                val_tex+=float(texloss)
                val_loss= float(chromaloss)+float(constloss)+float(wbloss)

                # Fake
                pred_fake= dc_model(pred_shd_val.detach())
                loss_D_fake = torch.mean(pred_fake)
                # Real
                pred_real= dc_model(shd_val_gt)
                loss_D_real = torch.mean(pred_real.detach())
                W_D = loss_D_fake - loss_D_real
                loss_bm_gan = torch.mean(pred_fake)
                val_wd+=float(W_D)
                val_gan+=float(loss_bm_gan)

        val_chroma=val_chroma/len(valloader)
        val_const=val_const/len(valloader)
        val_loss=val_loss/len(valloader)
        val_wb=val_wb/len(valloader)
        val_wd=val_wd/len(valloader)
        val_tex=val_tex/len(valloader)
        val_gan=val_gan/len(valloader)

        if args.logtb:
            idxs=torch.LongTensor(6).random_(0, wbs_val.shape[0])
            grid_alb_pred = torchvision.utils.make_grid(pred_alb_val[idxs],normalize=True, scale_each=True)
            grid_wbs_gt = torchvision.utils.make_grid(wbs_val[idxs],normalize=True, scale_each=True)
            grid_wbs_pred = torchvision.utils.make_grid(pred_wbs_val[idxs],normalize=True, scale_each=True)
            grid_shd_pred = torchvision.utils.make_grid(pred_shd_val[idxs],normalize=True, scale_each=True)
            grid_shd_gt = torchvision.utils.make_grid(shd_val_gt[idxs],normalize=True, scale_each=True)
            grid_tex_pred = torchvision.utils.make_grid(pred_tex_val[idxs],normalize=True, scale_each=True)
            grid_tex_gt = torchvision.utils.make_grid(albs_val[idxs],normalize=True, scale_each=True)
            writer.add_image('alb_pred/val', grid_alb_pred, global_step)
            writer.add_image('wb_inp/val', grid_wbs_gt, global_step)
            writer.add_image('wb_pred/val', grid_wbs_pred, global_step)
            writer.add_image('shd_gt/val', grid_shd_gt, global_step)
            writer.add_image('shd_pred/val', grid_shd_pred, global_step)
            writer.add_image('tex_gt/train', grid_tex_gt, global_step)
            writer.add_image('tex_pred/train', grid_tex_pred, global_step)
            writer.add_scalar('Loss/val', float(val_loss), global_step)
            writer.add_scalar('CLoss/val', float(val_chroma), global_step)
            writer.add_scalar('CnLoss/val', float(val_const), global_step)
            writer.add_scalar('WbLoss/val', float(val_wb), global_step)
            writer.add_scalar('TexLoss/val', float(val_tex), global_step)
            writer.add_scalar('gen_loss/val', float(val_gan), global_step)
            writer.add_scalar('em_dist/val', float(val_wd), global_step)
        print("Validation Chroma Loss:'{}'".format(val_chroma))
        sched.step(val_loss)
        dc_sched.step(val_wd)

        if val_loss < best_val_loss:
            best_val_loss=val_loss
            state = {'epoch': epoch,'model_state': model.state_dict()}
            torch.save(state, logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(arch,epoch,val_loss,train_loss,experiment_name))
        if val_wd < best_val_wd:
            best_val_wd=val_wd
            state = {'epoch': epoch,'model_state': dc_model.state_dict()}
            torch.save(state, logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(dc_arch,epoch,val_wd,train_wd,experiment_name))
        
        if (epoch % 5)==0:
            state = {'epoch': epoch,'model_state': model.state_dict()}
            torch.save(state, logdir+"{}_{}_{}_{}_{}_model.pkl".format(arch,epoch,val_loss,train_loss,experiment_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--imgsize', nargs='?', type=int, default=256, help='image size')
    parser.add_argument('--epochs', nargs='?', type=int, default=100, help='num of epochs')
    parser.add_argument('--batch', nargs='?', type=int, default=50, help='batch size')
    parser.add_argument('--resume', nargs='?', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--dc_resume', nargs='?', type=str, default=None, help='Path to the discriminator checkpoint')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--l_rate_dc', nargs='?', type=float, default=0.00001, help='Learning rate of the Discriminator')
    parser.add_argument('--logtb', nargs='?', type=bool, default=False, help='use tensorboard')

    args = parser.parse_args()

    #model=ImgCamNet(args)
    #print model
    train(args)