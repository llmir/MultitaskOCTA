import torch
import os
import time
import logging
import random
import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import create_train_arg_parser, build_model, define_loss, AverageMeter, generate_dataset
from sklearn.metrics import cohen_kappa_score


def train(epoch, model, data_loader, optimizer, criterion, device, training=False):
    seg_losses = AverageMeter("Seg_Loss", ".16f")
    dices = AverageMeter("Dice", ".8f")
    jaccards = AverageMeter("Jaccard", ".8f")
    clas_losses = AverageMeter("Clas_Loss", ".16f")
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

        outputs, label = model(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        preds = torch.round(outputs[0])

        seg_criterion, dice_criterion, jaccard_criterion, clas_criterion = criterion[0], criterion[1], criterion[2], criterion[3]

        seg_loss = seg_criterion(outputs[0], targets[0].to(torch.float32))
        dice = 1 - dice_criterion(preds.squeeze(1), targets[0].squeeze(1))
        jaccard = 1 - jaccard_criterion(preds.squeeze(1), targets[0].squeeze(1))

        clf_labels = torch.argmax(targets[3], dim=2).squeeze(1)
        clf_preds = torch.argmax(label, dim=1)
        clas_loss = clas_criterion(label, clf_labels)
        predictions = clf_preds.detach().cpu().numpy()
        target = clf_labels.detach().cpu().numpy()
        acc = (predictions == target).sum() / len(predictions)
        kappa = cohen_kappa_score(predictions, target)

        if training:
            if epoch < 30:
                total_loss = seg_loss
            else:
                total_loss = seg_loss + clas_loss
            total_loss.backward()
        optimizer.step()

        seg_losses.update(seg_loss.item(), inputs.size(0))
        dices.update(dice.item(), inputs.size(0))
        jaccards.update(jaccard.item(), inputs.size(0))
        clas_losses.update(clas_loss.item(), inputs.size(0))
        accs.update(acc.item(), inputs.size(0))
        kappas.update(kappa.item(), inputs.size(0))

        process.set_description('Seg Loss: ' + str(round(seg_losses.avg, 4)) + ' Clas Loss: ' + str(round(clas_losses.avg, 4)))

    epoch_seg_loss = seg_losses.avg
    epoch_dice = dices.avg
    epoch_jaccard = jaccards.avg
    epoch_clas_loss = clas_losses.avg
    epoch_acc = accs.avg
    epoch_kappa = kappas.avg

    return epoch_seg_loss, epoch_dice, epoch_jaccard, epoch_clas_loss, epoch_acc, epoch_kappa


def main():

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
    print(args)

    train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
    random.shuffle(train_file_names)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    if args.pretrain in ['imagenet', 'ssl', 'swsl']:
        pretrain = args.pretrain
        model = build_model(args.model_type, args.encoder, pretrain, aux=True)
    else:
        pretrain = None
        model = build_model(args.model_type, args.encoder, pretrain, aux=True)
    logging.info(model)
    model = model.to(device)

    train_loader, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, args.batch_size, args.distance_type, args.clahe)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    criterion = [
        define_loss(args.loss_type),
        smp.utils.losses.DiceLoss(),
        smp.utils.losses.JaccardLoss(),
        smp.utils.losses.CrossEntropyLoss()
    ]

    max_dice = 0
    max_acc = 0
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))

        train_seg_loss, train_dice, train_jaccard, train_clas_loss, train_acc, train_kappa = train(epoch, model, train_loader, optimizer, criterion, device, training=True)
        val_seg_loss, val_dice, val_jaccard, val_clas_loss, val_acc, val_kappa = train(epoch, model, valid_loader, optimizer, criterion, device, training=False)

        epoch_info = "Epoch: {}".format(epoch)
        train_seg_info = "Training Seg Loss:    {:.4f}, Training Dice:   {:.4f}, Training Jaccard:   {:.4f}".format(train_seg_loss, train_dice, train_jaccard)
        train_clas_info = "Training Clas Loss:   {:.4f}, Training Acc:    {:.4f}, Training Kappa:     {:.4f}".format(train_clas_loss, train_acc, train_kappa)
        val_seg_info = "Validation Seg Loss:  {:.4f}, Validation Dice: {:.4f}, Validation Jaccard: {:.4f}".format(val_seg_loss, val_dice, val_jaccard)
        val_clas_info = "Validation Clas Loss: {:.4f}, Validation Acc:  {:.4f}, Validation Kappa:   {:.4f}".format(val_clas_loss, val_acc, val_kappa)
        print(train_seg_info)
        print(train_clas_info)
        print(val_seg_info)
        print(val_clas_info)
        logging.info(epoch_info)
        logging.info(train_seg_info)
        logging.info(train_clas_info)
        logging.info(val_seg_info)
        logging.info(val_clas_info)
        writer.add_scalar("train_seg_loss", train_seg_loss, epoch)
        writer.add_scalar("train_dice", train_dice, epoch)
        writer.add_scalar("train_jaccard", train_jaccard, epoch)
        writer.add_scalar("train_clas_loss", train_clas_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_kappa", train_kappa, epoch)
        writer.add_scalar("val_seg_loss", val_seg_loss, epoch)
        writer.add_scalar("val_dice", val_dice, epoch)
        writer.add_scalar("val_jaccard", val_jaccard, epoch)
        writer.add_scalar("val_clas_loss", val_clas_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_kappa", val_kappa, epoch)

        best_name = os.path.join(args.save_path, "best_dice_" + str(round(val_dice, 4)) + "_jaccard_" + str(round(val_jaccard, 4)) + "_acc_" + str(round(val_acc, 4)) + "_kappa_" + str(round(val_kappa, 4)) + ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "_dice_" + str(round(val_dice, 4)) + "_jaccard_" + str(round(val_jaccard, 4)) + "_acc_" + str(round(val_acc, 4)) + "_kappa_" + str(round(val_kappa, 4)) + ".pt")

        if max_dice < val_dice:
            max_dice = val_dice
            if max_dice > 0.5:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg model saved!')
                logging.warning('Best seg model saved!')
        if max_acc < val_acc:
            max_acc = val_acc
            if max_acc > 0.4:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best clas model saved!')
                logging.warning('Best clas model saved!')
        if epoch % 50 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))
            else:
                torch.save(model.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))


if __name__ == "__main__":
    main()
