import torch
import os
from torch.utils.data import DataLoader
from dataset import DatasetImageMaskContourDist, mean_and_std
import glob
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser
import scipy.io as scio
from utils import AverageMeter
import segmentation_models_pytorch as smp
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, recall_score, f1_score, classification_report, jaccard_score
from train_seg_clf import CotrainingModelMulti
import pandas as pd
from scipy.special import softmax
import surface_distance
import scipy.spatial
from numpy import mean


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getJaccard(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.jaccard(testArray, resultArray)


def getPrecisionAndRecall(testImage, resultImage):
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    TP = np.sum(testArray*resultArray)
    FP = np.sum((1-testArray)*resultArray)
    FN = np.sum(testArray*(1-resultArray))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return precision, recall


def build_model(model_type):

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

    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()

    test_path = os.path.join(args.test_path, "*.png")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type
    distance_type = args.distance_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
    train_mean, train_std = mean_and_std(train_file_names)
    test_file_names = glob.glob(test_path)
    test_dataset = DatasetImageMaskContourDist(test_file_names, distance_type, train_mean, train_std, args.clahe)
    testLoader = DataLoader(test_dataset, batch_size=4, num_workers=4, shuffle=True)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    clf_accs = AverageMeter("Acc", ".8f")
    clf_kappas = AverageMeter("Kappa", ".8f")

    encoder = args.encoder
    attention_type = args.attention
    if args.pretrain in ['imagenet', 'ssl', 'swsl', 'instagram']:
        pretrain = args.pretrain
    else:
        pretrain = None
    usenorm = args.usenorm
    print("clahe:", args.clahe)
    model = CotrainingModelMulti(encoder, pretrain, usenorm, attention_type, args.classnum).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    name = []
    prob = []
    label = []
    pred = []
    dice_1o = []
    dice_2o = []
    jaccard_1o = []
    jaccard_2o = []
    HD_o = []
    ASSD_o = []

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(testLoader)):

        inputs = inputs.to(device)
        seg_labels = targets1.numpy()
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        seg_outputs = model.seg_forward(inputs)
        if not isinstance(seg_outputs, list):
            seg_outputs = [seg_outputs]

        clf_outputs = model.clf_forward(inputs, seg_outputs[3], seg_outputs[4], seg_outputs[5])
        outputs1 = seg_outputs[0].detach().cpu().numpy().squeeze()
        outputs2 = seg_outputs[1].detach().cpu().numpy().squeeze()
        outputs3 = seg_outputs[2].detach().cpu().numpy().squeeze()
        seg_preds = np.round(outputs1)

        clf_labels = torch.argmax(targets[3], dim=2).squeeze(1).detach().cpu().item()
        clf_preds = torch.argmax(clf_outputs, dim=1).detach().cpu().numpy().item()

        dsc_loss = smp.utils.losses.DiceLoss()
        jac_loss = smp.utils.losses.JaccardLoss()
        seg_prs = seg_preds
        dice_1 = f1_score(seg_labels.squeeze(), seg_prs, average='micro')
        dice_2 = getDSC(seg_labels, seg_prs)
        jaccard_1 = jaccard_score(seg_labels.squeeze(), seg_prs, average='micro')
        jaccard_2 = getJaccard(seg_labels, seg_prs)

        label_seg = np.array(seg_labels.squeeze(), dtype=bool)
        predict = np.array(seg_preds, dtype=bool)

        surface_distances = surface_distance.compute_surface_distances(label_seg, predict, spacing_mm=(1, 1))

        HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]

        ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

        output_path_m = os.path.join(
            save_path, "m_" + os.path.basename(img_file_name[0])
        )
        output_path_d = os.path.join(
            save_path, "d_" + os.path.basename(img_file_name[0])
        )
        output_path_dmat = os.path.join(
            save_path, "d_" + os.path.basename(img_file_name[0]).replace('.png', '.mat')
        )
        output_path_p = os.path.join(
            save_path, os.path.basename(img_file_name[0])
        )
        output_path_b = os.path.join(
            save_path, "b_" + os.path.basename(img_file_name[0])
        )
        output_path_bmat = os.path.join(
            save_path, "b_" + os.path.basename(img_file_name[0]).replace('.png', '.mat')
        )

        cv2.imwrite(output_path_p, (outputs1*255.))
        cv2.imwrite(output_path_m, (seg_preds*255.))
        cv2.imwrite(output_path_b, (outputs2*255.))
        cv2.imwrite(output_path_d, (outputs3*255.))
        scio.savemat(output_path_bmat, {'boundary': outputs2})
        scio.savemat(output_path_dmat, {'dist': outputs3})
        name.append(os.path.basename(img_file_name[0]))
        prob.append(softmax(clf_outputs.detach().cpu().numpy().squeeze()))
        label.append(clf_labels)
        pred.append(clf_preds)
        dice_1o.append(dice_1)
        dice_2o.append(dice_2)
        jaccard_1o.append(jaccard_1)
        jaccard_2o.append(jaccard_2)
        HD_o.append(HD)
        ASSD_o.append(ASSD)

    kappa = cohen_kappa_score(label, pred)
    acc = accuracy_score(label, pred)
    recall = recall_score(label, pred, average='micro')
    f1 = f1_score(label, pred, average='weighted')
    c_matrix = confusion_matrix(label, pred)
    if args.classnum == 3:
        target_names = ['N', 'D', 'M']
        clas_report = classification_report(label, pred, target_names=target_names, digits=5)
    elif args.classnum == 2:
        target_names = ['N', 'D']
        clas_report = classification_report(label, pred, target_names=target_names, digits=5)

    name_flag = args.val_path[11:12].replace('/', '_')
    print(name_flag)
    dataframe = pd.DataFrame({'case': name, 'prob': prob, 'label': label, 'pred': pred, 'dice1': dice_1o, 'dice2': dice_2o, 'jaccard1': jaccard_1o, 'jaccard2': jaccard_2o, 'HD': HD_o, 'ASSD': ASSD_o})
    dataframe.to_csv(save_path + "/" + name_flag + "_class&seg.csv", index=False, sep=',')
    resultframe = pd.DataFrame({'acc': acc, 'kappa': kappa, 'recall': recall, 'f1score': f1, 'seg_dice1': mean(dice_1o), 'seg_dice2': mean(dice_2o), 'jaccard1': mean(jaccard_1o), 'jaccard2': mean(jaccard_2o), 'HD': mean(HD_o), 'ASSD': mean(ASSD_o)}, index=[1])
    resultframe.to_csv(save_path + "/" + name_flag + "_acc_kappa.csv", index=0)
    with open(save_path + "/" + name_flag + "_cmatrix.txt", "w") as f:
        f.write(str(c_matrix))
    with open(save_path + "/" + name_flag + "_clas_report.txt", "w") as f:
        f.write(str(clas_report))
