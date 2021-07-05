# -*- coding: utf-8 -*-
import time
from smp_model_basic import MyUnetModel, my_get_encoder
import torch
import os
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import random
from tensorboardX import SummaryWriter
from utils import create_train_arg_parser, define_loss
import segmentation_models_pytorch as smp
import numpy as np
from sklearn.metrics import cohen_kappa_score
from utils import generate_dataset

IN_MODELS = ['unet_smp', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def segmentation_iteration(model, optimizer, model_type, criterion, data_loader, device, writer, training=False):
    running_loss = 0.0
    total_size = 0

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)

        if model_type in IN_MODELS + ["unet"]:
            if not isinstance(outputs, list):
                outputs = [outputs]
            loss = criterion(outputs[0], targets[0])
            dsc_loss = smp.utils.losses.DiceLoss()
            # 只写了一个例子
            # preds = torch.argmax(outputs[0].exp(), dim=1)
            preds = torch.argmax(torch.sigmoid(outputs[0]), dim=1)
            dsc = 1 - dsc_loss(preds, targets[0].squeeze(1))

        elif model_type == "dcan":
            loss = criterion(outputs[0], outputs[1], targets[0], targets[1])

        elif model_type == "dmtn":
            loss = criterion(outputs[0], outputs[1], targets[0], targets[2])

        elif model_type in ["psinet", "convmcd"]:
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
            )
        else:
            raise ValueError('error')

        if training:
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        total_size += inputs.size(0)

    epoch_loss = running_loss / total_size
    # print("total size:", total_size, training)

    return epoch_loss, dsc


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


def seg_clf_iteration(epoch, model, optimizer, criterion, data_loader, device, writer, loss_weights, startpoint, training=False):
    seg_losses = AverageMeter("Loss", ".16f")
    seg_dices = AverageMeter("Dice", ".8f")
    seg_jaccards = AverageMeter("Jaccard", ".8f")
    clf_losses = AverageMeter("Loss", ".16f")
    clf_accs = AverageMeter("Acc", ".8f")
    clf_kappas = AverageMeter("Kappa", ".8f")

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        if training:
            optimizer.zero_grad()

        seg_outputs = model.seg_forward(inputs)
        # seg_preds = torch.argmax(seg_outputs[0].exp(), dim=1)
        if not isinstance(seg_outputs, list):
            seg_outputs = [seg_outputs]

        seg_preds = torch.round(seg_outputs[0])
        clf_outputs = model.clf_forward(inputs, seg_outputs[1])

        # print(seg_criterion)j
        # print(seg_outputs[0].shape, targets[0].shape)
        seg_criterion, dice_criterion, jaccard_criterion, clf_criterion = criterion[0], criterion[1], criterion[2], criterion[3]
        seg_loss = seg_criterion(seg_outputs[0], targets[0].to(torch.float32))
        # print(seg_preds.shape, targets[0].squeeze(1).shape)
        seg_dice = 1 - dice_criterion(seg_preds.squeeze(1), targets[0].squeeze(1))
        seg_jaccard = 1 - jaccard_criterion(seg_preds.squeeze(1), targets[0].squeeze(1))
        # seg_iou = smp.utils.metrics.IoU(threshold=0.5)

        # print(clf_outputs.shape, targets[3].shape)
        clf_labels = torch.argmax(targets[3], dim=2).squeeze(1)
        clf_preds = torch.argmax(clf_outputs, dim=1)
        clf_loss = clf_criterion(clf_outputs, clf_labels)
        kappa = cohen_kappa_score(clf_preds.detach().cpu().numpy(), clf_labels.detach().cpu().numpy())
        # print(targets[3])
        # print(clf_labels)
        # print(clf_preds)
        acc = np.mean(clf_labels.detach().cpu().numpy() == clf_preds.detach().cpu().numpy())

        if training:
            if epoch <= startpoint:
                loss = seg_loss
            else:
                loss = (seg_loss + clf_loss)
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            # scheduler.step()

        seg_losses.update(seg_loss.item(), inputs.size(0))
        seg_dices.update(seg_dice.item(), inputs.size(0))
        seg_jaccards.update(seg_jaccard.item(), inputs.size(0))
        clf_losses.update(clf_loss.item(), inputs.size(0))
        clf_accs.update(acc, inputs.size(0))
        clf_kappas.update(kappa, inputs.size(0))

    seg_epoch_loss = seg_losses.avg
    seg_epoch_dice = seg_dices.avg
    seg_epoch_jaccard = seg_jaccards.avg
    clf_epoch_loss = clf_losses.avg
    clf_epoch_acc = clf_accs.avg
    clf_epoch_kappa = clf_kappas.avg
    # clf_epoch_loss = clf_running_loss / total_size
    # print("total size:", total_size, training, seg_epoch_loss)

    return seg_epoch_loss, seg_epoch_dice, seg_epoch_jaccard, clf_epoch_loss, clf_epoch_acc, clf_epoch_kappa


class CotrainingModel(nn.Module):

    def __init__(self, encoder, pretrain, usenorm, attention_type, classnum):
        super().__init__()
        self.seg_model = MyUnetModel(
            encoder_name=encoder, encoder_depth=5, encoder_weights=pretrain, decoder_use_batchnorm=usenorm,
            decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=attention_type, in_channels=1, classes=1,
            activation='sigmoid', aux_params=None
        )
        self.clf_model = my_get_encoder(encoder, in_channels=1, depth=5, weights=pretrain, num_classes=classnum)

    def seg_forward(self, x):
        return self.seg_model(x)

    def clf_forward(self, x, decoder_features):
        return self.clf_model(x, decoder_features)


def main():
    with torch.backends.cudnn.flags(enabled=True, benchmark=True,           deterministic=False, allow_tf32=False):
        torch.set_num_threads(4)
        set_seed(2021)

        args = create_train_arg_parser().parse_args()
        CUDA_SELECT = "cuda:{}".format(args.cuda_no)

        log_path = os.path.join(args.save_path, "summary/")
        writer = SummaryWriter(log_dir=log_path)
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_name = os.path.join(log_path, str(rq) + '.log')
        logging.basicConfig(
            filename=log_name,
            filemode="a",
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M",
            level=logging.INFO,
        )
        logging.info(args)

        device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

        encoder = args.encoder
        usenorm = args.usenorm
        attention_type = args.attention
        if args.pretrain in ['imagenet', 'ssl', 'swsl']:
            pretrain = args.pretrain
            # preprocess_input = get_preprocessing_fn(encoder, pretrain)
        else:
            pretrain = None
            # preprocess_input = get_preprocessing_fn(encoder)
        model = CotrainingModel(encoder, pretrain, usenorm, attention_type, args.classnum).to(device)
        logging.info(model)
        # seg_criterion = smp.utils.losses.DiceLoss()
        # seg_dice_criterion = smp.utils.losses.DiceLoss()
        # clf_criterion = smp.utils.losses.CrossEntropyLoss()
        criterion = [
            define_loss(args.loss_type),
            smp.utils.losses.DiceLoss(),
            smp.utils.losses.JaccardLoss(),
            smp.utils.losses.CrossEntropyLoss()
        ]

        optimizer = Adam([
            {"params": model.seg_model.parameters(), "lr": args.LR_seg},
            {"params": model.clf_model.parameters(), "lr": args.LR_clf}
        ])

        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
        random.shuffle(train_file_names)
        val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))

        # train_dataset = DatasetImageMaskContourDist(train_file_names, args.distance_type)
        # valid_dataset = DatasetImageMaskContourDist(val_file_names, args.distance_type)
        # trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
        # devLoader = DataLoader(valid_dataset, batch_size=args.val_batch_size, num_workers=4)
        trainLoader, devLoader = generate_dataset(train_file_names, val_file_names, args.batch_size, args.val_batch_size, args.distance_type, args.clahe)

        epoch_start = 0
        max_dice = 0.8
        max_acc = 0.6
        loss_weights = [1, 1, 1]
        startpoint = args.startpoint

        for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

            print('\nEpoch: {}'.format(epoch))
            training_seg_loss, training_seg_dice, training_seg_jaccard, training_clf_loss, training_clf_acc, training_clf_kappa = seg_clf_iteration(epoch, model, optimizer, criterion, trainLoader, device, writer, loss_weights, startpoint, training=True)
            dev_seg_loss, dev_seg_dice, dev_seg_jaccard, dev_clf_loss, dev_clf_acc, dev_clf_kappa = seg_clf_iteration(epoch, model, optimizer, criterion, devLoader, device, writer, loss_weights, startpoint, training=False)

            epoch_info = "Epoch: {}".format(epoch)
            train_info = "TrainSeg Loss:{:.7f}, Dice: {:.7f}, Jaccard: {:.7f}, TrainClf Loss:{:.7f}, Acc: {:.7f}, Kappa:{:.7f}".format(training_seg_loss, training_seg_dice, training_seg_jaccard, training_clf_loss, training_clf_acc, training_clf_kappa)
            val_info = "ValSeg Loss:{:.7f}, Dice: {:.7f}, Jaccard: {:.7f}, ValClf Loss:{:.7f}, Acc: {:.7f}, Kappa:{:.7f}:".format(dev_seg_loss, dev_seg_dice, dev_seg_jaccard, dev_clf_loss, dev_clf_acc, dev_clf_kappa)
            print(train_info)
            print(val_info)
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            writer.add_scalar("trainseg_loss", training_seg_loss, epoch)
            writer.add_scalar("trainseg_dice", training_seg_dice, epoch)
            writer.add_scalar("trainseg_jaccard", training_seg_jaccard, epoch)
            writer.add_scalar("traincls_loss", training_clf_loss, epoch)
            writer.add_scalar("traincls_acc", training_clf_acc, epoch)
            writer.add_scalar("traincls_kappa", training_clf_kappa, epoch)

            writer.add_scalar("valseg_loss", dev_seg_loss, epoch)
            writer.add_scalar("valseg_dice", dev_seg_dice, epoch)
            writer.add_scalar("valseg_jaccard", dev_seg_jaccard, epoch)
            writer.add_scalar("valcls_loss", dev_clf_loss, epoch)
            writer.add_scalar("valcls_acc", dev_clf_acc, epoch)
            writer.add_scalar("valcls_kappa", dev_clf_kappa, epoch)

            best_name = os.path.join(args.save_path, "dice_" + str(round(dev_seg_dice, 5)) + "_jaccard_" + str(round(dev_seg_jaccard, 5)) + "_acc_" + str(round(dev_clf_acc, 4)) + "_kap_" + str(round(dev_clf_kappa, 4)) + ".pt")
            save_name = os.path.join(args.save_path, str(epoch) + "_dice_" + str(round(dev_seg_dice, 5)) + "_jaccard_" + str(round(dev_seg_jaccard, 5)) + "_acc_" + str(round(dev_clf_acc, 4)) + "_kap_" + str(round(dev_clf_kappa, 4)) + ".pt")

            if max_dice <= dev_seg_dice:
                max_dice = dev_seg_dice
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg model saved!')
                logging.warning('Best seg model saved!')
            if max_acc <= dev_clf_acc:
                max_acc = dev_clf_acc
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best clf model saved!')
                logging.warning('Best clf model saved!')

            if epoch % 50 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), save_name)
                    print('Epoch {} model saved!'.format(epoch))
                else:
                    torch.save(model.state_dict(), save_name)
                    print('Epoch {} model saved!'.format(epoch))


if __name__ == "__main__":
    main()
