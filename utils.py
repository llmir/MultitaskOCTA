import torch
import numpy as np
import torchvision
import time
import argparse
import segmentation_models_pytorch as smp
from tqdm import tqdm
from typing import List
from torch import Tensor, einsum
from torch.nn import functional as F
from dataset import DatasetImageMaskContourDist, mean_and_std
from dataset import DatasetCornea, distancedStainingImage
from losses import LossUNet, LossDCAN, LossDMTN, LossPsiNet
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def build_model(model_type, encoder, pretrain, aux=False):

    aux_params = dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=3,                 # define number of output labels
    )

    if model_type == "unet":
        model = UNet(num_classes=2)
    if model_type == "dcan":
        model = UNet_DCAN(num_classes=2)
    if model_type == "dmtn":
        model = UNet_DMTN(num_classes=2)
    if model_type == "psinet":
        model = PsiNet(num_classes=2)
    if model_type == "convmcd":
        model = UNet_ConvMCD(num_classes=2)
    if model_type == "unet_smp":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels=3,
            classes=1,
            activation='sigmoid',
            aux_params=None if not aux else aux_params
        )
    if model_type == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels=3,
            classes=1,
            activation='sigmoid',
            aux_params=None if not aux else aux_params
        )
    if model_type == "manet":
        model = smp.MAnet(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_pab_channels=64,
            in_channels=1,
            classes=1,
            activation='sigmoid',
            aux_params=None if not aux else aux_params
        )
    if model_type == "linknet":
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            in_channels=1,
            classes=1,
            activation='sigmoid',
            aux_params=None if not aux else aux_params
        )
    if model_type == "fpn":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_dropout=0.2,
            in_channels=3,
            classes=1,
            activation='sigmoid',
            upsampling=4,
            aux_params=None if not aux else aux_params
        )
    if model_type == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=pretrain,
            encoder_depth=3,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=3,
            classes=1,
            activation='sigmoid',
            upsampling=8,
            aux_params=None if not aux else aux_params
        )
    if model_type == "pan":
        model = smp.PAN(
            encoder_name=encoder,
            encoder_weights=pretrain,
            encoder_dilation=True,
            decoder_channels=32,
            in_channels=1,
            classes=1,
            activation='sigmoid',
            upsampling=4,
            aux_params=None if not aux else aux_params
        )
    if model_type == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_channels=256,
            in_channels=3,
            classes=1,
            activation='sigmoid',
            upsampling=8,
            aux_params=None if not aux else aux_params
        )
    if model_type == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=1,
            classes=1,
            activation='sigmoid',
            upsampling=4,
            aux_params=None if not aux else aux_params
        )
    # print(model)

    return model


def define_loss(loss_type, weights=[3, 1, 2]):

    if loss_type == "jaccard":
        criterion = smp.utils.losses.JaccardLoss()
    if loss_type == "dice":
        criterion = smp.utils.losses.DiceLoss()
    if loss_type == "ce":
        criterion = smp.utils.losses.CrossEntropyLoss()
    if loss_type == "bcewithlogit":
        criterion = smp.utils.losses.BCEWithLogitsLoss()
    if loss_type == "unet":
        criterion = LossUNet(weights)
    if loss_type == "dcan":
        criterion = LossDCAN(weights)
    if loss_type == "dmtn":
        criterion = LossDMTN(weights)
    if loss_type == "psinet" or loss_type == "convmcd":
        # Both psinet and convmcd uses same mask,contour and distance loss function
        criterion = LossPsiNet(weights)

    return criterion


def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    dsces = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, _, _, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.nll_loss(outputs[0], targets.squeeze(1))
            dsc_loss = smp.utils.losses.DiceLoss()

            preds = torch.argmax(outputs[0].exp(), dim=1)
            dsc = 1 - dsc_loss(preds, targets.squeeze(1))

            losses.append(loss.item())
            dsces.append(dsc.item())

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

    return np.mean(losses), np.mean(dsces), time.perf_counter() - start


def evaluate_modi(device, epoch, model, data_loader, writer, criterion, model_type):
    model.eval()
    losses = []
    dsces = []
    start = time.perf_counter()

    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, _, _, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]
            preds = torch.argmax(outputs[0].exp(), dim=1)
            loss = criterion(preds, targets.squeeze(1))

            dsc_loss = smp.utils.losses.DiceLoss()

            dsc = 1 - dsc_loss(preds, targets.squeeze(1))
            losses.append(loss.item())
            dsces.append(dsc.item())

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

    return np.mean(losses), np.mean(dsces), time.perf_counter() - start


def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=False):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _, _, _ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)

            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train", val_batch_size)
                save_image(output_final, "Prediction_train", val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def generate_dataset(train_file_names, val_file_names, batch_size, val_batch_size, distance_type, do_clahe):
    train_mean, train_std = mean_and_std(train_file_names)

    train_dataset = DatasetImageMaskContourDist(train_file_names, distance_type, train_mean, train_std, do_clahe)
    valid_dataset = DatasetImageMaskContourDist(val_file_names, distance_type, train_mean, train_std, do_clahe)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size, num_workers=4, shuffle=True)

    return train_loader, valid_loader


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to training img jpg files")
    parser.add_argument("--val_path", type=str, help="path to validation img jpg files")
    parser.add_argument("--test_path", type=str, help="path to test img jpg files")
    parser.add_argument(
        "--train_type",
        type=str,
        default="cotraining",
        help="Select training type, including single classification, segmentation, cotraining and multitask. ")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: unet,dcan,dmtn,psinet,convmcd",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_mask",
        help="select distance transform type - dist_mask,dist_contour,dist_signed",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=64, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")
    parser.add_argument("--encoder", type=str, default=None, help="encoder.")
    parser.add_argument("--pretrain", type=str, default=None, help="choose pretrain.")
    parser.add_argument("--loss_type", type=str, default=None, help="loss type.")
    parser.add_argument("--local_rank", default=0, type=int, help='node rank for distributed training')
    parser.add_argument("--LR_seg", default=1e-4, type=float, help='learning rate.')
    parser.add_argument("--LR_clf", default=1e-4, type=float, help='learning rate.')
    parser.add_argument("--use_scheduler", type=bool, default=False, help="use_scheduler.")
    parser.add_argument("--aux", type=bool, default=False, help="choose to do classification")
    parser.add_argument("--attention", type=str, default=None, help="decoder_attention_type.")
    parser.add_argument("--usenorm", type=bool, default=True, help="encoder use bn")
    parser.add_argument("--startpoint", type=int, default=60, help="start cotraining point.")
    parser.add_argument("--clahe", type=bool, default=False, help="do clahe.")
    parser.add_argument("--classnum", type=int, default=3, help="clf class number.")
    parser.add_argument("--fold", type=str, default=0, help="Fold for training.")
    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: unet,dcan,dmtn,psinet,convmcd",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_signed",
        help="select distance transform type - dist_mask,dist_contour,dist_signed",
    )
    parser.add_argument("--train_path", type=str, help="path to train img jpg files")
    parser.add_argument("--val_path", type=str, help="path to validation img jpg files")
    parser.add_argument("--test_path", type=str, help="path to test img jpg files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument("--encoder", type=str, default=None, help="encoder.")
    parser.add_argument("--pretrain", type=str, default=None, help="choose pretrain.")
    parser.add_argument("--attention", type=str, default=None, help="decoder_attention_type.")
    parser.add_argument("--val_batch_size", type=int, default=32, help="validation batch size")
    parser.add_argument("--usenorm", type=bool, default=True, help="encoder use bn")
    parser.add_argument("--clahe", type=bool, default=False, help="do clahe.")
    parser.add_argument("--classnum", type=int, default=3, help="clf class number.")
    return parser
