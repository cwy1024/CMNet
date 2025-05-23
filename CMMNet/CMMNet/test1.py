import os
import time
import math
import glob
import torch
import random
import joblib
from PIL import Image

import losses
import imageio
import argparse
import warnings
import numpy as np
import torchvision
from tqdm import tqdm
import torch.nn as nn
import sklearn.externals
from BraDataSet import *
import torch.optim as optim
from datetime import datetime
from utils import count_params
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from skimage.io import imread, imsave
from torch.utils.data import DataLoader
from hausdorff import hausdorff_distance
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
from metrics import dice_coef, batch_iou, mean_iou, iou_score, ppv, sensitivity

from CMNet.CMMNet.Mamba_cnn.CMNet import *

from train import test_img_paths, test_mask_paths


# IMG_PATH = glob.glob(r"F:\CwyDataSet\Brats\Brats2018\MICCAI_BraTS_2018_Data_Training\processed\2D\testImage\*")
# MASK_PATH = glob.glob(r"F:\CwyDataSet\Brats\Brats2018\MICCAI_BraTS_2018_Data_Training\processed\2D\testMask\*")

# GetPicture,Calculate
MODE = 'GetPicture'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='CMNet',
                        help='model name')
    parser.add_argument('--mode', default=MODE,
                        help='GetPicture or Calculate')
    parser.add_argument('--dataset', default="Brats2020",
                        help='dataset name')
    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/%s/information.pkl' % (val_args.name, val_args.dataset))

    if not os.path.exists('output3/%s' % args.name):
        os.makedirs('output3/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    # create model
    print("=> creating model %s" % args.name)

    model=Mamba_Cnn()
    model = model.cuda()

    # Data loading code
    # img_paths = IMG_PATH
    # mask_paths = MASK_PATH

    test_img = test_img_paths
    test_mask = test_mask_paths

    # train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #   train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/%s/model.pth' % (args.name, args.dataset)))
    model.eval()

    val_dataset = BraDataset(args, test_img, test_mask)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if val_args.mode == "GetPicture":

        val_pred_path = 'output3/%s/' % args.name + "pred/"
        if not os.path.exists(val_pred_path):
            os.mkdir(val_pred_path)

    if val_args.mode == "GetPicture":

        """
        获取并保存模型生成的标签图
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.  cuda()
                    # target = target.cuda()

                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                    # print("img_paths[i]:%s" % img_paths[i])
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = test_img[args.batch_size * i:args.batch_size * (i + 1)]
                    # print("output_shape:%s"%str(output.shape))

                    for i in range(output.shape[0]):
                        """
                        生成灰色圖片
                        wtName = os.path.basename(img_paths[i])
                        overNum = wtName.find(".npy")
                        wtName = wtName[0:overNum]
                        wtName = wtName + "_WT" + ".png"
                        imsave('output/%s/'%args.name + wtName, (output[i,0,:,:]*255).astype('uint8'))
                        tcName = os.path.basename(img_paths[i])
                        overNum = tcName.find(".npy")
                        tcName = tcName[0:overNum]
                        tcName = tcName + "_TC" + ".png"
                        imsave('output/%s/'%args.name + tcName, (output[i,1,:,:]*255).astype('uint8'))
                        etName = os.path.basename(img_paths[i])
                        overNum = etName.find(".npy")
                        etName = etName[0:overNum]
                        etName = etName + "_ET" + ".png"
                        imsave('output/%s/'%args.name + etName, (output[i,2,:,:]*255).astype('uint8'))
                        """
                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName + ".png"
                        rgbPic = np.zeros([224, 224, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i, 0, idx, idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i, 1, idx, idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i, 2, idx, idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                        # imsave('output3/%s/pred/' % args.name + rgbName, rgbPic)
                        img_pil = Image.fromarray(rgbPic)
                        img_pil.save('output3/%s/pred/%s' % (args.name, rgbName), dpi=(600, 600))  # 设置DPI为600

            torch.cuda.empty_cache()
        """
        将验证集中的GT numpy格式转换成图片格式并保存
        """
        print("\nSaving GT,numpy to picture")
        val_gt_path = 'output3/%s/' % args.name + "GT/"
        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(test_mask))):
            mask_path = test_mask[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"

            npmask = np.load(mask_path)

            GtColor = np.zeros([npmask.shape[0], npmask.shape[1], 3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):
                    # 坏疽(NET,non-enhancing tumor)(标签1) 红色
                    if npmask[idx, idy] == 1:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0
                    # 浮肿区域(ED,peritumoral edema) (标签2) 绿色
                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 128
                        GtColor[idx, idy, 2] = 0
                    # 增强肿瘤区域(ET,enhancing tumor)(标签4) 黄色
                    elif npmask[idx, idy] == 4:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0
            # imageio.imwrite(val_gt_path + rgbName, GtColor)
            # 读取图像
            img = Image.fromarray(GtColor)
            # 保存图像并设置dpi
            img.save(val_gt_path + rgbName, dpi=(600, 600))
            """
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            wtName = name + "_WT" + ".png"
            tcName = name + "_TC" + ".png"
            etName = name + "_ET" + ".png"

            npmask = np.load(mask_path)

            WT_Label = npmask.copy()
            WT_Label[npmask == 1] = 1.
            WT_Label[npmask == 2] = 1.
            WT_Label[npmask == 4] = 1.
            TC_Label = npmask.copy()
            TC_Label[npmask == 1] = 1.
            TC_Label[npmask == 2] = 0.
            TC_Label[npmask == 4] = 1.
            ET_Label = npmask.copy()
            ET_Label[npmask == 1] = 0.
            ET_Label[npmask == 2] = 0.
            ET_Label[npmask == 4] = 1.

            imsave(val_gt_path + wtName, (WT_Label * 255).astype('uint8'))
            imsave(val_gt_path + tcName, (TC_Label * 255).astype('uint8'))
            imsave(val_gt_path + etName, (ET_Label * 255).astype('uint8'))
            """
        print("Done!")

    if val_args.mode == "Calculate":
        """
        计算各种指标:Dice、Sensitivity、PPV
        """
        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []

        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []

        maskPath = glob("output3/%s/GT/*.png" % args.name)
        pbPath = glob("output3/%s/pred/*.png" % args.name)
        if len(maskPath) == 0:
            print("请先生成图片!")
            return

        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    # 只要这个像素的任何一个通道有值,就代表这个像素不属于前景,即属于WT区域
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    # 只要第一个通道是255,即可判断是TC区域,因为红色和黄色的第一个通道都是255,区别于绿色
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    # 只要第二个通道是128,即可判断是ET区域
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
            # 开始计算WT
            dice = dice_coef(wtpbregion, wtmaskregion)
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
            # 开始计算TC
            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
            # 开始计算ET
            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)

        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("==============================")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("==============================")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("==============================")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("==============================")


if __name__ == '__main__':
    main()
