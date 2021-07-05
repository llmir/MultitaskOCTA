import torch
import os
from torch.utils.data import DataLoader
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import random
from tensorboardX import SummaryWriter
from utils import visualize, evaluate, create_train_arg_parser, evaluate_modi
from losses import LossUNet, LossDCAN, LossDMTN, LossPsiNet
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from dataset import DatasetImageMaskContourDist
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from apex import amp


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True

# SEED = 2021
# utils.set_global_seed(SEED)
# utils.prepare_cudnn(deterministic=True)

def build_model(model_type, encoder, pretrain):

    if model_type == "unet":
        model = UNet(num_classes=2)
        print(model)
    if model_type == "dcan":
        model = UNet_DCAN(num_classes=2)
        print(model)
    if model_type == "dmtn":
        model = UNet_DMTN(num_classes=2)
        print(model)
    if model_type == "psinet":
        model = PsiNet(num_classes=2)
        print(model)
    if model_type == "convmcd":
        model = UNet_ConvMCD(num_classes=2)
        print(model)
    if model_type == "unet_smp":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels=1,
            classes=1,
            activation=None,
            aux_params=None
        )
        print(model)
    if model_type == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None,
            in_channels=1,
            classes=1,
            activation=None,
            aux_params=None
        )
        print(model)
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
            activation=None,
            aux_params=None
        )
        print(model)
    if model_type == "linknet":
        model = smp.Linknet(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            in_channels=1,
            classes=1,
            activation=None,
            aux_params=None)
        print(model)
    if model_type == "fpn":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_dropout=0.2,
            in_channels=1,
            classes=1,
            activation=None,
            upsampling=4,
            aux_params=None
            )
        print(model)
    if model_type == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=pretrain,
            encoder_depth=3,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=1,
            classes=1,
            activation=None,
            upsampling=8,
            aux_params=None
            )
        print(model)
    if model_type == "pan":
        model = smp.PAN(
            encoder_name=encoder,
            encoder_weights=pretrain,
            encoder_dilation=True,
            decoder_channels=32,
            in_channels=1,
            classes=1,
            activation=None,
            upsampling=4,
            aux_params=None
            )
        print(model)
    if model_type == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            decoder_channels=256,
            in_channels=1,
            classes=1,
            activation=None,
            upsampling=8,
            aux_params=None
            )
        print(model)
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
            activation=None,
            upsampling=4,
            aux_params=None
            )
        print(model)
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


def train_model(model, targets, model_type, criterion, optimizer):
    in_models = ['unet_smp', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']

    if model_type in in_models:

        optimizer.zero_grad()
        outputs = model(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        # print('\n', np.array(outputs).shape)
        # print('\n', np.array(targets).shape)
        loss = criterion(outputs[0], targets[0])
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # scheduler.step()

    elif model_type == "unet":
        optimizer.zero_grad()
        outputs = model(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]

        loss = criterion(outputs[0], targets[0])
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # scheduler.step()

    elif model_type == "dcan":

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0], outputs[1], targets[0], targets[1])
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # scheduler.step()

    elif model_type == "dmtn":

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0], outputs[1], targets[0], targets[2])
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # scheduler.step()

    elif model_type == "psinet" or model_type == "convmcd":

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(
                        outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
                    )
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # scheduler.step()

    else:
        print('error')

    return loss

# ====================================================


if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()
    encoder = args.encoder
    if args.pretrain in ['imagenet', 'ssl', 'swsl']:
        pretrain = args.pretrain
        preprocess_input = get_preprocessing_fn(encoder, pretrain)
    else:
        pretrain = None
        # preprocess_input = get_preprocessing_fn(encoder) ##
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)

    logging.basicConfig(
        filename=''.format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")

    train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
    random.shuffle(train_file_names)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    if args.pretrain in ['imagenet', 'ssl', 'swsl']:
        model = build_model(args.model_type, args.encoder, pretrain)
    else:
        pretrain = None
        model = build_model(args.model_type, args.encoder, pretrain)

    model = model.to(device)

    optimizer = Adam(model.parameters(), args.LR)

    criterion = define_loss(args.loss_type)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 400], gamma=0.2)    ###
    if args.use_scheduler is True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    in_models = ['unet_smp', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']

    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)

    torch.set_num_threads(2)

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(train_file_names, args.distance_type),
        batch_size=args.batch_size, num_workers=4
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names, args.distance_type), num_workers=4
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names, args.distance_type),
        batch_size=args.val_batch_size, num_workers=4
    )

    for epoch in range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        model.train()

        for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(
            tqdm(trainLoader)
        ):

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)
            targets4 = targets4.to(device)

            targets = [targets1, targets2, targets3, targets4]

            loss = train_model(model, targets, args.model_type, criterion, optimizer)

            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_file_names)

        if epoch % 1 == 0:
            if args.model_type not in in_models:
                dev_loss, dsc, dev_time = evaluate(device, epoch, model, devLoader, writer)
            else:
                dev_loss, dsc, dev_time = evaluate_modi(device, epoch, model, devLoader, writer, criterion, args.model_type)
            writer.add_scalar("loss_valid", dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            print("Global Loss:{} Val Loss:{} dsc:{}".format(epoch_loss, dev_loss, dsc))
        else:
            print("Global Loss:{} ".format(epoch_loss))

        if args.use_scheduler is True:
            scheduler.step(dev_loss)

        logging.info("epoch:{} train_loss:{}".format(epoch, epoch_loss))

        if epoch % 25 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.save_path, str(epoch) + "_val_" + str(round(dev_loss, 5)) + "_dsc_" + str(round(dsc, 5)) + ".pt")
                )
            else:
                torch.save(
                    model.state_dict(), os.path.join(args.save_path, str(epoch) + "_val_" + str(round(dev_loss, 5)) + "_dsc_" + str(round(dsc, 5)) + ".pt")
                )
