import copy
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_iou, check_results

def create_PR_curve(preds,gts):
    # return generated PR points AP50
    AP = 0.5
    data = []
    truth = []
    # recompose the prediction and sort by scores
    for box_i,class_i,score_i in zip(preds['boxes'],preds['classes'],preds['scores']):
        data.append([box_i,class_i,score_i]) 
    data = sorted(data,key=lambda k:k[:][-1],reverse=True)
    # recompose the ground truth 
    for box_j,class_j in zip(gts['boxes'],gts['classes']):
        truth.append([box_j,class_j])
    # compare prediction to ground truth and calculate PR data
    PR = []
    TP = 0
    detected = False
    for index_i, prediction in enumerate(data):
        for index_j, groundtruth in enumerate(truth):
            if calculate_iou(prediction[0],groundtruth[0]) > AP:
                detected = True
                if prediction[1] == groundtruth[1]:
                    TP+=1
        num_pred = index_i+1
        num_truth = len(truth)
        precision = TP/num_pred
        recall = TP/num_truth
        PR.append([precision,recall])
    print(PR)
    
    # create smoothed PR
    PR = sorted(PR, key=lambda k:k[:][1])
    sorted_PR = PR.copy()
    P_max = 0
    for i,data_i in enumerate(PR):
        for j,data_j in enumerate(PR[i+1::]):
            if sorted_PR[i][0] <= data_j[0]:
                sorted_PR[i][0] = data_j[0]
    print(sorted_PR)
    return PR

def calculate_mAP(PR):
    X = 0
    mAP = 0
    for index, data in enumerate(PR):
        mAP = mAP + data[0]*(data[1]-X)
        X = data[1]
    print(mAP)
    return mAP

if __name__ == '__main__':
    # load data 
    with open('L6-16/data/predictions.json', 'r') as f:
        preds = json.load(f)

    with open('L6-16/data/ground_truths.json', 'r') as f:
        gts = json.load(f)
    
    # TODO IMPLEMENT THIS SCRIPT
    PR = create_PR_curve(preds[0],gts[0])
    mAP = calculate_mAP(PR)
    check_results(mAP)