import json
# import numpy as np
from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    filtered = []
    # IMPLEMENT THIS FUNCTION
    for i,box_i in enumerate(predictions["boxes"]):
        for j,box_j in enumerate(predictions["boxes"][i:]):
            print(len(predictions["boxes"][i:]))
            print(calculate_iou(box_i,box_j))

    return filtered


if __name__ == '__main__':
    with open('L6-12/data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
    check_results(filtered)