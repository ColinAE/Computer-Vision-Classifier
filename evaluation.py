# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:58:03 2016

@author: haiming
"""

import sys
import numpy as np
import copy
threashold = 0.5
frame_c_r = 0
frame_c_w = 0
# gt.txt the text file that contain ground truth
# dt.txt the text file that contain the region of interest created by our algorithm
def print_Usage_and_Exit():
    # threashold from 0.1 to 0.5
    print('Usage: evaluation.py (gt.txt) (dt.txt) (threadshold)')
    sys.exit(-1)

def load_file(gt_file, dt_file):
    def bs(bytes):
        '''Convert bytes to string and remove double quotes'''
        return str(bytes,'utf-8').replace('"','')
    classification = {'car':0, 'person':1, 'motorcycle':2,'unknown':10}
    converters = {9: lambda x: classification[bs(x)]}
    datagt = np.loadtxt(gt_file, delimiter=' ', converters=converters, dtype='int')
    datadt = np.loadtxt(dt_file, delimiter=' ', converters=converters, dtype='int')
    #sort the data by frame number                    
    datagt_sort = datagt[datagt[:, 5].argsort()]
    datadt_sort = datadt[datadt[:, 5].argsort()]
    # delete all the objects that out of frame
    datagt_com = datagt_sort[datagt_sort[:, 6]==0]
    datadt_com = datadt #datadt[datadt[:, 6]==0]
    return (datagt_com, datadt_com)

def frame_class(dt, gt):
    global frame_c_r
    global frame_c_w
    if dt[9] == gt[9]:
        frame_c_r += 1
    else:
        frame_c_w += 1
        
#simple algorithm to find the TP TN and FP
def FindOverLap(gts, dts):
    dt_list = []   
    if gts.shape[0] != 0 and dts.shape[0] != 0:
        for gt in gts:
            GTArea = (gt[3] - gt[1])*(gt[4]-gt[2])
            for dt in dts:
                DTArea = (dt[3] - dt[1])*(dt[4]-dt[2])
                cross_xmin = np.maximum(gt[1], dt[1])
                cross_ymin = np.maximum(gt[2], dt[2])
                cross_xmax = np.minimum(gt[3], dt[3])
                cross_ymax = np.minimum(gt[4], dt[4])
                if cross_xmin >= cross_xmax or cross_ymin >= cross_ymax:
                    cross_area = 0
                else:
                    cross_area = (cross_xmax - cross_xmin) * (cross_ymax - cross_ymin)                    
                overlap_percentage = 2 * cross_area / (GTArea + DTArea)
                if overlap_percentage >= threashold:
                    dt_list.append([dt[0], gt[0]])
                    frame_class(dt, gt)
                # print("GT id %d, DT id %d, cross_area %f" % (gt[0], dt[0], overlap_percentage))
    return [(dts.shape[0], gts.shape[0]), dt_list]

def frame_based_detection(gt_data, dt_data):
    first_frame = dt_data[0, 5] - 1
    last_frame = dt_data[-1, 5] - 1
    # print(first_frame, last_frame)
    full_list = []
    for frame in range(first_frame, last_frame + 1):
       # print(frame)
        gts = gt_data[gt_data[:, 5] == frame]
        dts = dt_data[dt_data[:, 5] == frame + 1]
        full_list.append([frame, FindOverLap(gts, dts)])
        pair_dict = {}
        #Frame based calculation
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for struct in full_list:
        frame, dt_list = struct
        (n_dt, n_gt), pairs = dt_list
        dt_v = np.zeros(1000, dtype = int)
        gt_v = np.zeros(1000, dtype = int)
        if len(pairs) != 0: 
            for pair in pairs:
                dt_id, gt_id = pair
                dt_v[dt_id] = 1
                gt_v[gt_id] = 1
                index = (dt_id, gt_id)
                if index not in pair_dict:
                    pair_dict[index] = [frame]
                else:
                   pair_dict[index].append(frame)
        TP = np.sum(gt_v)
        FP = n_dt - np.sum(dt_v)
        FN = n_gt - TP
        TP_sum += TP
        FP_sum += FP 
        FN_sum += FN 
        # print(frame, TP, FP, FN)
    print("Frame level:TP=%d FP=%d FN=%d" % (TP_sum, FP_sum, FN_sum))
    sensitivity = TP_sum/(TP_sum + FN_sum)
    PPV = TP_sum/(TP_sum + FP_sum)
    print("Frame Level: S=%.3f PPV=%.3f" %(sensitivity, PPV))
    print("Frame Level: R=%d W=%d %.3f" %(frame_c_r, frame_c_w, frame_c_r/(frame_c_r+frame_c_w)))
    return pair_dict

def gt_analysis(gt_frame, dt_frame, pair_dict):
    gt_r = copy.copy(gt_frame)
    FP_frame = 0
    TP_frame = 0
    FN_frame = 0   
    FN_P = 0
    for key,value in pair_dict.items():
        if len(value) >= 20:
            dt_index, gt_index = key
            gt_r[gt_index] = list(set(gt_r[gt_index]) - set(value))
            FP_frame += len(list(set(dt_frame[dt_index]) - set(value)))
    i = 0
    for key,value in gt_r.items():
        # print(key, len(value) / len(gt_frame[key]))
        TP_frame += len(gt_frame[key]) - len(value)
        FN_frame += len(value)
        p1 = len(value) / len(gt_frame[key])
        FN_P += p1
        i += 1
    s = TP_frame / (TP_frame + FN_frame)
    ppv = TP_frame / (TP_frame + FP_frame)
    print("object level, ground truth")
    print(TP_frame, FP_frame, FN_frame, s, ppv)
    print(FN_P/i, 1-(FN_P/i))
    
def dt_analysis(gt_frame, dt_frame, pair_dict):
    dt_r = copy.copy(dt_frame)
    FP_frame = 0
    TP_frame = 0
    FN_frame = 0
    FP_P = 0
    for key,value in pair_dict.items():
        if len(value) >= 20:
            dt_index, gt_index = key
            dt_r[dt_index] = list(set(dt_r[dt_index]) - set(value))
            FN_frame += len(list(set(gt_frame[gt_index]) - set(value)))
    i = 0          
    for key,value in dt_r.items():
        # print(key, len(value))
        TP_frame += len(dt_frame[key]) - len(value)
        FP_frame += len(value)
        p1 = len(value) / len(dt_frame[key])
        FP_P += p1
        i += 1
    s = TP_frame / (TP_frame + FN_frame)
    ppv = TP_frame / (TP_frame + FP_frame)
    print("object level, system measurement")
    print(TP_frame, FP_frame, FN_frame, s, ppv)
    print(FP_P/i, 1-(FP_P/i))
    
def match_class(pair_dict, datagt_com, datadt_com):
    right_c = 0
    wrong_c = 0
    for key,value in pair_dict.items():
        if len(value) >= 20:
            dt_index, gt_index = key
            for frame in value:
                gt_c = datagt_com[datagt_com[:, 5] == frame]
                gt_c = gt_c[gt_c[:,0] == gt_index][0, 9]
                dt_c = datadt_com[datadt_com[:, 5] == frame+1]
                dt_c = dt_c[dt_c[:,0] == dt_index][0, 9]
                if gt_c == dt_c:
                    right_c += 1
                else:
                    wrong_c += 1
    percentage = right_c/(right_c + wrong_c)
    print("object level, classification")
    print(right_c, wrong_c, percentage)

def object_based_detection(pair_dict, gt_data, dt_data):
    gt_frame = {}
    dt_frame = {}
    #Objects based calculation
    first_frame = dt_data[0, 5] - 1
    last_frame = dt_data[-1, 5] - 1
    # print(first_frame, last_frame)
    for frame in range(first_frame, last_frame + 1):
        #print(frame)
        gts = gt_data[gt_data[:, 5] == frame]
        dts = dt_data[dt_data[:, 5] == frame + 1]
        if gts.shape[0] != 0:
            for gt in gts:
                if gt[0] not in gt_frame:
                    gt_frame[gt[0]] = [frame]
                else:
                    gt_frame[gt[0]].append(frame)
        if dts.shape[0] != 0:
            for dt in dts:
                if dt[0] not in dt_frame:
                    dt_frame[dt[0]] = [frame]
                else:
                    dt_frame[dt[0]].append(frame)
    gt_analysis(gt_frame, dt_frame, pair_dict)
    dt_analysis(gt_frame, dt_frame, pair_dict)
    match_class(pair_dict, gt_data, dt_data)
    
def main(arg=None):
    if len(sys.argv) != 4:
        print_Usage_and_Exit()
    gt_file = sys.argv[1]
    dt_file = sys.argv[2]
    global threashold
    threashold = float(sys.argv[3])
    gt_data, dt_data = load_file(gt_file, dt_file)
    p_d = frame_based_detection(gt_data, dt_data)
    object_based_detection(p_d, gt_data, dt_data)
    return 0

if __name__ == '__main__':
    main()