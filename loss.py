import torch 
import torch.nn as nn
import torch.nn.functional as F
import kornia
from math import pi
from utils import *

"""
Loss Functions
"""
def chromaticity_loss(pred_wbk,target_wb, input_img, msk):
	eps=1e-5
	pred_wb=pred_wbk*input_img
	# target_wb=target_wbk*input_img
	pred_wb_int=torch.sum(pred_wb, keepdim=True, dim=1)
	inp_img_int=torch.sum(input_img*msk, keepdim=True, dim=1)

	pred_chroma=pred_wb/(pred_wb_int+eps)
	target_chroma=target_wb/(torch.sum(target_wb, keepdim=True, dim=1)+eps)

	pred_chroma=pred_chroma*msk
	target_chroma=target_chroma*msk

	closs=F.l1_loss(pred_chroma,target_chroma, reduction='sum')/torch.sum(msk)
	iloss=F.l1_loss(pred_wb_int, inp_img_int, reduction='sum')/torch.sum(msk)
	loss = iloss+closs

	return loss, pred_wb


def chromaticity_wb_mse(pred_wb, wb, msk):
	eps=1e-5
	pred_wb_chroma=torch.div(pred_wb,torch.sum(pred_wb, keepdim=True, dim=1)+eps)
	wb_chroma=torch.div(wb,torch.sum(wb, keepdim=True, dim=1)+eps)
	pred_wb_chroma=pred_wb_chroma*msk
	wb_chroma=wb_chroma*msk
	loss=F.mse_loss(pred_wb_chroma,wb_chroma, reduction='sum')/torch.sum(msk)
	return loss


def chromaticity_mat_mse(pred_mat_alb, pred_mat, wb, msk):
	eps=1e-5
	n,c,h,w=pred_mat_alb.shape
	# pred_mat_alb_chroma=torch.div(pred_mat_alb,torch.sum(pred_mat_alb, keepdim=True, dim=1)+eps)
	# wb_chroma=torch.div(wb,torch.sum(wb, keepdim=True, dim=1)+eps)
	pred_mat_alb_chroma=torch_masked_divide(pred_mat_alb,torch.sum(pred_mat_alb, keepdim=True, dim=1))
	wb_chroma=torch_masked_divide(wb,torch.sum(wb, keepdim=True, dim=1))
	pred_mat_alb_chroma=pred_mat_alb_chroma*msk
	wb_chroma=wb_chroma*msk
	closs=F.l1_loss(pred_mat_alb_chroma,wb_chroma, reduction='sum')/torch.sum(msk)

	# penalize the L1 norm of the gradients for the albedo 
	pred_mat_gray = kornia.rgb_to_grayscale(pred_mat)
	pred_grads= kornia.spatial_gradient(pred_mat_gray, order=1, normalized=True).view(n,-1)
	# l2pen=torch.norm(pred_grads, p=2)/(n*h*w)
	l1pen=torch.norm(pred_grads, p=1)/(n*h*w)

	loss=closs+5.0*l1pen
	return loss

def scale_inv_loss(pred, target, msk, lmbda=0.5):
	d=target-pred
	n=torch.sum(msk)
	losst1= (1/n)*torch.sum(torch.pow(d,2))
	losst2= lmbda*(1/(n**2))*torch.pow(torch.sum(d),2)

	loss=losst1-losst2
	return loss

def log_scale_inv_loss(pred, target, msk, lmbda=0.5):
	d=torch.log1p(target*msk)-torch.log1p(pred*msk)
	n=torch.sum(msk)
	losst1= (1/n)*torch.sum(torch.pow(d,2))
	losst2= lmbda*(1/(n**2))*torch.pow(torch.sum(d),2)

	loss=losst1-losst2
	return loss

def grad_loss(pred_wbk, target_wbk, reduction='mean', invalid_msk=None):
	pred_wbk_gray = kornia.rgb_to_grayscale(pred_wbk)
	pred_grads= kornia.spatial_gradient(pred_wbk_gray, order=1, normalized=True).squeeze(1)
	pred_gradx = pred_grads[:,0]
	pred_grady = pred_grads[:,1]
	target_wbk_gray = kornia.rgb_to_grayscale(target_wbk)
	target_grads= kornia.spatial_gradient(target_wbk_gray, order=1, normalized=True).squeeze(1)
	target_gradx = target_grads[:,0]
	target_grady = target_grads[:,1]

	if reduction=='mean':
		loss = F.l1_loss(pred_gradx,target_gradx)+F.l1_loss(pred_grady,target_grady)
	else:
		loss = F.l1_loss(pred_gradx*invalid_msk,target_gradx*invalid_msk, reduction=reduction)/torch.sum(invalid_msk)+F.l1_loss(pred_grady*invalid_msk,target_grady*invalid_msk, reduction=reduction)/torch.sum(invalid_msk)
	return loss, target_grads

def edge_l1(pred, target):
	pred_gray = kornia.rgb_to_grayscale(pred)
	pred_grads= kornia.spatial_gradient(pred_gray, order=1, normalized=True).squeeze(1)
	target_gray = kornia.rgb_to_grayscale(target)
	target_grads= kornia.spatial_gradient(target_gray, order=1, normalized=True).squeeze(1)
	loss = F.l1_loss(pred_grads,target_grads)

	return loss

def shading_l1_penalty(pred_shd):
	n,c,h,w=pred_shd.shape
	pred_grads= kornia.spatial_gradient(pred_shd, order=2, normalized=True).view(n,-1)
	l1pen=torch.norm(pred_grads, p=1)/(n*h*w)
	return l1pen

def angular_loss(pred, target):        
	"""
	https://github.com/acecreamu/angularGAN
	"""
	# ACOS
	cos_between = torch.nn.CosineSimilarity(dim=1)
	cos = cos_between(target, pred)
	cos = torch.clamp(cos,-0.99999, 0.99999)
	loss = torch.mean(torch.acos(cos))

	# MSE
	# loss = torch.mean((illum_gt - illum_pred)**2)

	# 1 - COS
	# loss = 1 - torch.mean(cos)

	# 1 - COS^2
	# loss = 1 - torch.mean(cos**2)
	return loss
