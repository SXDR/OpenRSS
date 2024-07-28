# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 
 
# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette_MF():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    person = [64, 64, 0]
    bike = [0, 128, 192]
    curve = [0, 0, 192]
    car_stop = [128, 128, 0]
    guardrail = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump = [192, 64, 0]
    palette = np.array([unlabelled, car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette

def get_palette_MCubeS():
    # road = [128, 64, 128]
    # person = [220, 20, 60]
    # car = [0, 0, 142]
    # bicycle = [119, 11, 32]
    # building = [70, 70, 70]
    # wall = [102, 102, 156]
    # bridge = [70, 130, 180]
    # pole = [153, 153, 153]
    # terrain = [152, 251, 152]
    # unlabelled = [0, 0, 0]
    road = [152, 255, 152]
    person = [250, 250, 210]
    car = [230, 230, 250]
    bicycle = [176, 196, 222]
    building = [255, 255, 153]

    wall = [173, 216, 230]
    bridge = [255, 204, 204]

    # pole = [230, 230, 250]
    pole = [221, 163, 222]
    terrain = [255, 182, 193]
    unlabelled = [238, 172, 137]
    palette = np.array(
        [road, person, car, bicycle, building, wall, bridge, pole, terrain, unlabelled])
    return palette


def get_palette_KP():
    road = [128, 64, 128]
    sidewalk = [244, 35, 232]
    building = [70, 70, 70]
    wall = [102, 102, 156]
    fence = [190, 153, 153]
    pole = [153, 153, 153]
    traffic_light = [250, 170, 30]
    traffic_sign = [220, 220, 0]
    vegetation = [107, 142, 35]
    terrain = [152, 251, 152]
    sky = [70, 130, 180]
    person = [220, 20, 60]
    rider = [255, 0, 0]
    car = [0, 0, 142]
    truck = [0, 0, 70]
    bus = [0, 60, 100]
    train = [0, 80, 100]
    motorcycle = [0, 0, 230]
    bicycle = [119, 11, 32]
    unlabelled = [0, 0, 0]
    palette = np.array(
        [road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign, vegetation, terrain, sky, person,
         rider, car, truck, bus, train, motorcycle, bicycle, unlabelled])
    return palette


def get_palette_PST():
    unlabelled = [0, 0, 0]
    fire_extinguisher = [0, 0, 255]
    backpack = [0, 255, 0]
    hand_drill = [255, 0, 0]
    survivor = [255, 255, 255]
    palette = np.array([unlabelled, fire_extinguisher, backpack, hand_drill, survivor])
    return palette


def visualize(image_name, predictions, weight_name, data_name):
    palette = eval(data_name)()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('./result/Pred/tovs/'+ data_name + "/"+ weight_name + '_' + image_name[i] + '.png')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
