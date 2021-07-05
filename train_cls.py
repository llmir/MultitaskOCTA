import torch
import os
import time
import logging
import random
import glob
import numpy as np
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import create_train_arg_parser, AverageMeter, generate_dataset
from sklearn.metrics import cohen_kappa_score
from resnest.torch import resnest50


net_config = {
    'vgg16': models.vgg16_bn,
    'resnet50': models.resnet50,
    'resnext50': models.resnext50_32x4d,
    # 'resnest50': torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True),
    'resnest50': resnest50(pretrained=True),
    'args': {}
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class CustomizedModel(nn.Module):
    def __init__(self, name, backbone, num_classes, pretrained=False, **kwargs):
        super(CustomizedModel, self).__init__()

        if 'resnest' in name:
            net = resnest50(pretrained=True)
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = backbone(pretrained=pretrained, **kwargs)
        if 'resnet' in name or 'resnext' in name or 'shufflenet' in name:
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif 'densenet' in name:
            net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif 'vgg' in name:
            net.features = make_layers(cfgs['D'], batch_norm=True)
            net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        elif 'mobilenet' in name:
            net.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(net.last_channel, num_classes),
            )
        elif 'squeezenet' in name:
            net.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif 'resnest' in name:
            pass
        else:
            raise NotImplementedError('Not implemented network.')
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


def generate_model(network, out_features, net_config, device, pretrained=False, checkpoint=None):
    if pretrained:
        print('Loading weight from pretrained')
    if checkpoint:
        model = torch.load(checkpoint).to(device)
        print('Load weights form {}'.format(checkpoint))
    else:
        if network not in net_config.keys():
            raise NotImplementedError('Not implemented network.')

        model = CustomizedModel(
            network,
            net_config[network],
            out_features,
            pretrained,
            **net_config['args']
        ).to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, data_loader, optimizer, criterion, device, training=False):
    losses = AverageMeter("Cls_Loss", ".16f")
    accs = AverageMeter("Accuracy", ".8f")
    kappas = AverageMeter("Kappa", ".8f")

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    process = tqdm(data_loader)
    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(process):
        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)

        labels = torch.argmax(targets[3], dim=2).squeeze(1)
        preds = torch.argmax(outputs, dim=1)

        loss = criterion(outputs, labels)
        predictions = preds.detach().cpu().numpy()
        target = labels.detach().cpu().numpy()
        acc = (predictions == target).sum() / len(predictions)
        kappa = cohen_kappa_score(predictions, target)

        if training:
            loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc.item(), inputs.size(0))
        kappas.update(kappa.item(), inputs.size(0))

        process.set_description('Loss: ' + str(round(losses.avg, 4)))

    epoch_loss = losses.avg
    epoch_acc = accs.avg
    epoch_kappa = kappas.avg

    return epoch_loss, epoch_acc, epoch_kappa


def main():
    seed = 1234
    set_random_seed(seed)

    args = create_train_arg_parser().parse_args()
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)

    if args.pretrain == 'True':
        pretrain = True
    else:
        pretrain = False

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
    print(args)

    train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
    random.shuffle(train_file_names)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    model = generate_model(args.model_type, args.classnum, net_config, device, pretrained=pretrain, checkpoint=None)
    logging.info(model)
    model = model.to(device)

    train_loader, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, args.batch_size, args.distance_type, args.clahe)
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    criterion = smp.utils.losses.CrossEntropyLoss()

    max_acc = 0
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))

        train_loss, train_acc, train_kappa = train(model, train_loader, optimizer, criterion, device, training=True)
        val_loss, val_acc, val_kappa = train(model, valid_loader, optimizer, criterion, device, training=False)

        epoch_info = "Epoch: {}".format(epoch)
        train_info = "Training Loss:   {:.4f}, Training Acc:   {:.4f}, Training Kappa:   {:.4f}".format(train_loss, train_acc, train_kappa)
        val_info = "Validation Loss: {:.4f}, Validation Acc: {:.4f}, Validation Kappa: {:.4f}".format(val_loss, val_acc, val_kappa)
        print(train_info)
        print(val_info)
        logging.info(epoch_info)
        logging.info(train_info)
        logging.info(val_info)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_kappa", train_kappa, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_kappa", val_kappa, epoch)

        best_name = os.path.join(args.save_path, "best_acc_" + str(round(val_acc, 4)) + "_kappa_" + str(round(val_kappa, 4)) + ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "_acc_" + str(round(val_acc, 4)) + "_kappa_" + str(round(val_kappa, 4)) + ".pt")

        if max_acc < val_acc:
            max_acc = val_acc
            if max_acc > 0.3:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best model saved!')
                logging.warning('Best model saved!')
        if epoch % 50 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))
            else:
                torch.save(model.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))


if __name__ == "__main__":
    main()
