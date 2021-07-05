#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:40:54 2019

@author: wujon
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import surface_distance
# import nibabel as ni
import scipy.io
import scipy.spatial
import xlwt
import os
import cv2
from skimage import morphology
from skimage.morphology import thin
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score


os.chdir('./')


predictName = 'cotrain_192_pad'
predictPath = './smp/' + predictName + '/'
labelPath = "./smp/mask_ori_f1/"
name_experiment = 'exp_test'
path_experiment = './' + name_experiment + '/'

# labelPath = "./gt/"
# outpredictPath = "./gt_poor_o_thin/"
# outlabelPath = "./gt_o_thin/"


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


def intersection(testImage, resultImage):
    testSkel = morphology.skeletonize(testImage)
    testSkel = testSkel.astype(int)
    resultSkel = morphology.skeletonize(resultImage)
    resultSkel = resultSkel.astype(int)

    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    testSkel = testSkel.flatten()
    resultSkel = resultSkel.flatten()

    recall = np.sum(resultSkel * testArray) / (np.sum(testSkel))
    precision = np.sum(resultArray * testSkel) / (np.sum(testSkel))

    intersection = 2 * precision * recall / (precision + recall)
    return intersection


if __name__ == "__main__":
    labelList = os.listdir(labelPath)
    # labelList.sort(key = lambda x: int(x[:-4]))
    img_nums = len(labelList)

    Q1 = []
    Q2 = []
    Q3 = []
    Q4 = []
    Q5 = []
    Q6 = []
    Q7 = []
    Q8 = []
    Q9 = []
    Q10 = []
    Q11 = []
    Q12 = []
    Q13 = []
    Q14 = []
    Q15 = []
    Q16 = []
    Q17 = []

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    row_num = 0
    sheet.write(row_num, 0, 'CaseName')
    sheet.write(row_num, 1, 'DSC')
    sheet.write(row_num, 2, 'Pre')
    sheet.write(row_num, 3, 'Recall')
    sheet.write(row_num, 4, 'HD')
    sheet.write(row_num, 5, 'ASSD')
    sheet.write(row_num, 6, 'surface_dice_0')
    sheet.write(row_num, 7, 'rel_overlap_gt')
    sheet.write(row_num, 8, 'rel_overlap_pred')
    sheet.write(row_num, 9, 'intersec')
    sheet.write(row_num, 10, 'HD_thin')
    sheet.write(row_num, 11, 'ASSD_thin')
    sheet.write(row_num, 12, 'surface_dice_1')
    sheet.write(row_num, 13, 'surface_dice_2')
    sheet.write(row_num, 14, 'Jaccard')
    sheet.write(row_num, 15, 'acc')
    sheet.write(row_num, 16, 'spe')
    sheet.write(row_num, 17, 'sen')

    for idx, filename in enumerate(labelList):
        label = cv2.imread(labelPath + filename, 0)
        # print (label.dtype)

        # label = cv2.imread(labelPath + filename)
        label[label < 50] = 0
        label[label >= 50] = 1

        thinned_label = thin(label)
        # cv2.imwrite(outlabelPath+filename,(thinned_label*255).astype(np.uint8))
        # ret,label = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
        predict = cv2.imread(predictPath + filename.replace('_manual.png', '_expert.png'), 0)
        # print(predictPath + filename)
        # print (predict.dtype)
        # ret,predict = cv2.threshold(predict,127,255,cv2.THRESH_BINARY)
        # predict = cv2.imread(predictPath + filename)
        # predict = predict / 255
        predict[predict < 127] = 0
        predict[predict >= 127] = 1

# ==============================================================================================================================================================================
        y_scores = cv2.imread(predictPath + filename.replace('_manual.png', '_expert.png'), 0)  # #####################################################################
        y_scores = np.asarray(y_scores.flatten())/255.
        y_scores = y_scores[:, np.newaxis]
        # print(y_scores.shape)
        y_true = cv2.imread(labelPath + filename, 0)
        y_true = np.asarray(y_true.flatten())/255.

        # fpr, tpr, thresholds = roc_curve((y_true), y_scores)
        # AUC_ROC = roc_auc_score(y_true, y_scores)
        # # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        # print ("\nArea under the ROC curve: " +str(AUC_ROC))
        # roc_curve =plt.figure()
        # plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
        # plt.title('ROC curve')
        # plt.xlabel("FPR (False Positive Rate)")
        # plt.ylabel("TPR (True Positive Rate)")
        # plt.legend(loc="lower right")
        # plt.savefig(path_experiment+"ROC.png")
        # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
        # recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
        # AUC_prec_rec = np.trapz(precision,recall)
        # print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
        # prec_rec_curve = plt.figure()
        # plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
        # plt.title('Precision - Recall curve')
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend(loc="lower right")
        # plt.savefig(path_experiment+"Precision_recall.png")

        # def best_f1_threshold(precision, recall, thresholds):
        #     best_f1=-1
        #     for index in range(len(precision)):
        #         curr_f1=2.*precision[index]*recall[index]/(precision[index]+recall[index])
        #         if best_f1<curr_f1:
        #             best_f1=curr_f1
        #             best_threshold=thresholds[index]

        #     return best_f1, best_threshold

        # best_f1, best_threshold = best_f1_threshold(precision, recall, thresholds)
        # print("\nthresholds: " + str(thresholds))
        # print("\nbest_f1: " + str(best_f1))
        # print("\nbest_threshold: " + str(best_threshold))

        # Confusion matrix
        threshold_confusion = 0.5
        # print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
        y_pred = np.empty((y_scores.shape[0]))
        # print(y_scores.shape[0])
        # print(np.unique(y_pred))
        for i in range(y_scores.shape[0]):
            if y_scores[i] >= threshold_confusion:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        # print(np.unique(y_pred))
        # print(np.unique(y_true))
        confusion = confusion_matrix(y_true, y_pred)
        # print (confusion)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0]+confusion[1, 1])/float(np.sum(confusion))
        # print ("Global Accuracy: " +str(accuracy))
        specificity = 0
        if float(confusion[0, 0]+confusion[0, 1]) != 0:  # 00 tn   11 tp   10  fn  01  fp
            specificity = float(confusion[0, 0])/float(confusion[0, 0]+confusion[0, 1])
        # print ("Specificity: " +str(specificity))
        sensitivity = 0
        if float(confusion[1, 1]+confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1])/float(confusion[1, 1]+confusion[1, 0])
        # print ("Sensitivity: " +str(sensitivity))
        precision = 0
        if float(confusion[1, 1]+confusion[0, 1]) != 0:
            precision = float(confusion[1, 1])/float(confusion[1, 1]+confusion[0, 1])
        # print ("Precision: " +str(precision))

        if float(confusion[1, 1]+confusion[0, 1]) != 0:
            PPV = float(confusion[1, 1])/float(confusion[1, 1]+confusion[0, 1])
        # print ("PPV: " +str(PPV))

        # Jaccard similarity index
        jaccard_index = jaccard_score(y_true, y_pred)
        print("\nJaccard similarity score: " + str(jaccard_index))

        # F1 score
        F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
        # print ("\nF1 score (F-measure): " +str(F1_score))

        # Save the results
#         file_perf = open(path_experiment+'performances.txt', 'w')
#         # file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
#         #                 + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
#         #                 + "\nJaccard similarity score: " +str(jaccard_index)
#         #                 + "\nF1 score (F-measure): " +str(F1_score)
#         #                 +"\n\nConfusion matrix:"
#         #                 +str(confusion)
#         #                 +"\nACCURACY: " +str(accuracy)
#         #                 +"\nSENSITIVITY: " +str(sensitivity)
#         #                 +"\nSPECIFICITY: " +str(specificity)
#         #                 +"\nPRECISION: " +str(precision)
#         #                 +"\nRECALL: " +str(sensitivity)
#         #                 +"\nPPV: " +str(PPV)
#         #                 +"\nbest_th: " +str(best_threshold)
#         #                 +"\nbest_f1: " +str(best_f1)
#         #                 )
#         file_perf.write(
#                         "\nJaccard similarity score: " +str(jaccard_index)
#                         + "\nF1 score (F-measure): " +str(F1_score)
#                         +"\n\nConfusion matrix:"
#                         +str(confusion)
#                         +"\nACCURACY: " +str(accuracy)
#                         +"\nSENSITIVITY: " +str(sensitivity)
#                         +"\nSPECIFICITY: " +str(specificity)
#                         +"\nPRECISION: " +str(precision)
#                         +"\nRECALL: " +str(sensitivity)
#                         +"\nPPV: " +str(PPV)
#                         )
#         file_perf.close()
# #==============================================================================================================================================================================

        thinned_predict = thin(predict)
        # cv2.imwrite(outpredictPath+filename,(thinned_predict*255).astype(np.uint8))

        # predict[predict>=1] = 1
        # dice = getDSC(predict, label)
        # print("filename:" , filename , "dice:" , dice)
        # dice_res = "the " + filename[:-4] + " image's DSC : " + str(round(dice,4)) + "\n"
        DSC = getDSC(label, predict)
        # surface_distances = surface_distance.compute_surface_distances(label, predict, spacing_mm=(1, 1, 1))
        # HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        # distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        # distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        # surfel_areas_gt = surface_distances["surfel_areas_gt"]
        # surfel_areas_pred = surface_distances["surfel_areas_pred"]

        # ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) +np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))
        Jaccard = getJaccard(label, predict)

        precision, recall = getPrecisionAndRecall(label, predict)
        intersec = intersection(label, predict)

        label = np.array(label, dtype=bool)
        predict = np.array(predict, dtype=bool)

        surface_distances = surface_distance.compute_surface_distances(label, predict, spacing_mm=(1, 1))

        surface_distances_thin = surface_distance.compute_surface_distances(thinned_label, thinned_predict, spacing_mm=(1, 1))

        HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        HD_thin = surface_distance.compute_robust_hausdorff(surface_distances_thin, 95)

        surface_dice_2 = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 2)
        rel_overlap_gt, rel_overlap_pred = surface_distance.compute_surface_overlap_at_tolerance(surface_distances, 2)
        surface_dice_1 = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 1)
        surface_dice_0 = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 0)
        surface_dice_3 = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 3)

        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]

        ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

        distances_gt_to_pred_t = surface_distances_thin["distances_gt_to_pred"]
        distances_pred_to_gt_t = surface_distances_thin["distances_pred_to_gt"]
        surfel_areas_gt_t = surface_distances_thin["surfel_areas_gt"]
        surfel_areas_pred_t = surface_distances_thin["surfel_areas_pred"]

        ASSD_thin = (np.sum(distances_pred_to_gt_t * surfel_areas_pred_t) + np.sum(distances_gt_to_pred_t * surfel_areas_gt_t))/(np.sum(surfel_areas_gt_t)+np.sum(surfel_areas_pred_t))

        # print(surface_overlap)
        row_num += 1
        sheet.write(row_num, 0, filename)
        sheet.write(row_num, 1, DSC)
        sheet.write(row_num, 2, precision)
        sheet.write(row_num, 3, recall)
        sheet.write(row_num, 4, HD)
        sheet.write(row_num, 5, ASSD)
        sheet.write(row_num, 6, surface_dice_0)
        sheet.write(row_num, 7, rel_overlap_gt)
        sheet.write(row_num, 8, rel_overlap_pred)
        sheet.write(row_num, 9, intersec)
        sheet.write(row_num, 10, HD_thin)
        sheet.write(row_num, 11, ASSD_thin)
        sheet.write(row_num, 12, surface_dice_1)
        sheet.write(row_num, 13, surface_dice_2)
        # sheet.write(row_num, 14, surface_dice_3)
        sheet.write(row_num, 14, Jaccard)
        sheet.write(row_num, 15, accuracy)
        sheet.write(row_num, 16, specificity)
        sheet.write(row_num, 17, sensitivity)

        Q1.append(DSC)
        Q2.append(precision)
        Q3.append(recall)
        Q4.append(HD)
        Q5.append(ASSD)
        Q6.append(surface_dice_0)
        Q7.append(rel_overlap_gt)
        Q8.append(rel_overlap_pred)
        Q9.append(intersec)
        Q10.append(HD_thin)
        Q11.append(ASSD_thin)
        Q12.append(surface_dice_1)
        Q13.append(surface_dice_2)
        # Q14.append(surface_dice_3)
        Q14.append(Jaccard)
        Q15.append(accuracy)
        Q16.append(specificity)
        Q17.append(sensitivity)

    Q1 = np.array(Q1)
    Q2 = np.array(Q2)
    Q3 = np.array(Q3)
    Q4 = np.array(Q4)
    Q5 = np.array(Q5)
    Q6 = np.array(Q6)
    Q7 = np.array(Q7)
    Q8 = np.array(Q8)
    Q9 = np.array(Q9)
    Q10 = np.array(Q10)
    Q11 = np.array(Q11)
    Q12 = np.array(Q12)
    Q13 = np.array(Q13)
    Q14 = np.array(Q14)
    Q15 = np.array(Q15)
    Q16 = np.array(Q16)
    Q17 = np.array(Q17)

    row_num += 2
    sheet.write(row_num, 0, 'CaseName')
    sheet.write(row_num, 1, 'DSC')
    sheet.write(row_num, 2, 'Pre')
    sheet.write(row_num, 3, 'Recall')
    sheet.write(row_num, 4, 'HD')
    sheet.write(row_num, 5, 'ASSD')
    sheet.write(row_num, 6, 'surface_dice_0')
    sheet.write(row_num, 7, 'rel_overlap_gt')
    sheet.write(row_num, 8, 'rel_overlap_pred')
    sheet.write(row_num, 9, 'intersec')
    sheet.write(row_num, 10, 'HD_thin')
    sheet.write(row_num, 11, 'ASSD_thin')
    sheet.write(row_num, 12, 'surface_dice_1')
    sheet.write(row_num, 13, 'surface_dice_2')
    sheet.write(row_num, 14, 'Jaccard')
    sheet.write(row_num, 15, 'accuracy')
    sheet.write(row_num, 16, 'specificity')
    sheet.write(row_num, 17, 'sensitivity')

    row_num += 1
    sheet.write(row_num, 0, predictName)
    sheet.write(row_num, 1, Q1.mean())
    sheet.write(row_num, 2, Q2.mean())
    sheet.write(row_num, 3, Q3.mean())
    sheet.write(row_num, 4, Q4.mean())
    sheet.write(row_num, 5, Q5.mean())
    sheet.write(row_num, 6, Q6.mean())
    sheet.write(row_num, 7, Q7.mean())
    sheet.write(row_num, 8, Q8.mean())
    sheet.write(row_num, 9, Q9.mean())
    sheet.write(row_num, 10, Q10.mean())
    sheet.write(row_num, 11, Q11.mean())
    sheet.write(row_num, 12, Q12.mean())
    sheet.write(row_num, 13, Q13.mean())
    sheet.write(row_num, 14, Q14.mean())
    sheet.write(row_num, 15, Q15.mean())
    sheet.write(row_num, 16, Q16.mean())
    sheet.write(row_num, 17, Q17.mean())

    book.save('./smp/' + predictName + '.xls')
