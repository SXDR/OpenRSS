import os, argparse, time, datetime, sys, shutil, stat, torch
import sys

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.KP_dataset import KP_dataset
from util.PST_dataset import PST_dataset
from util.MCubeS_dataset import MCubeS_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from model import FEANet
from semseg.models import *
from semseg.models import segori
from segment_anything import sam_model_registry
import sam_lora_image_encoder
from segment_anything.utils import transforms
from segment_anything.utils.amg import build_all_layer_point_grids
import clip

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='openrss')
parser.add_argument('--weight_name', '-w', type=str, default='openrss')
parser.add_argument('--file_name', '-f', type=str, default='best.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=536)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=10)
parser.add_argument('--num_classes', '-ncs', type=int, default=8)
parser.add_argument('--data_dir', '-dr', type=str,
                    default='./dataset/multimodal_dataset') 
parser.add_argument('--model_dir', '-wd', type=str, default='./test/')
parser.add_argument('--vit_name', type=str,
                    default='vit_h', help='select one vit model')
parser.add_argument('--data_name', type=str,
                    default='get_palette_MCubeS')
parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth', 
                    help='Pretrained checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--img_size', type=int,
                    default=640, help='input patch size of network input')
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':
    class_num = ["road", "person", "car", "bicycle", "building", "wall", "bridge", "pole", "terrain", "Background"]
    text_list = []
    qz = "a pohot of "
    for i in class_num:
        text_list.append(qz + i)
    device_clip = "cpu"
    model_clip, preprocess = clip.load("./ViT-L-14.pt", device=device_clip)
    text = clip.tokenize(text_list).to(device_clip)

    with torch.no_grad():
        text_features = model_clip.encode_text(text)

    text_features = text_features.cuda(args.gpu)

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    # prepare save direcotry

    model_dir = args.model_dir
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    conf_total = np.zeros((args.n_class, args.n_class))
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size, num_classes=args.num_classes,
                                                                checkpoint=args.ckpt)

    model = sam_lora_image_encoder.LoRA_Sam(sam, args.rank)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)

    model.load_lora_parameters(model_file)
    print('done!')
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    transform = transforms.ResizeLongestSide(640)
    point_grids = build_all_layer_point_grids(32, 0, 1)
    cropped_im_size = (480, 640)
    points_scale = np.array(cropped_im_size)[None, ::-1]
    points_for_image = point_grids[0] * points_scale
    transformed_points = transform.apply_coords(points_for_image, cropped_im_size)
    in_points = torch.as_tensor(transformed_points, device=model.sam.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=model.sam.device)
    in_points = in_points[None, :, :]
    in_labels = in_labels[None, :]
    points = (in_points, in_labels)

    # missing_keys, unexpected_keys = model.backbone.load_state_dict(pretrained_weight, strict=False)

    batch_size = 1  # do not change this parameter!
    test_dataset = MCubeS_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_time = time.time()
            logits = model(images, points, text_features, multimask_output)
            logits = logits["masks"]
            end_time = time.time()
            if it >= 5:  # # ignore the first 5 frames
                ave_time_cost += (end_time - start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            # conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            conf = confusion_matrix(y_true=label, y_pred=prediction,
                                    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            conf_total += conf
            # save demo images
            if not os.path.exists('./result/' + 'Pred/' + args.weight_name + '/'):
                os.mkdir('./result/' + 'Pred/' + args.weight_name + '/')
            visualize(image_name=names, predictions=logits.argmax(1), weight_name='Pred_' + args.weight_name,
                      data_name=args.data_name)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  % (
                      args.model_name, args.weight_name, it + 1, len(test_loader), names,
                      (end_time - start_time) * 1000))

    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

    conf_total_matfile = os.path.join('./FEANet_coding/result/Pred_' + args.weight_name,
                                      'conf_' + args.weight_name + '.mat')
    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' % (
        args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' % (args.img_height, args.img_width))
    print('* the weight name: %s' % args.weight_name)
    print('* the file name: %s' % args.file_name)
    print(
        "* recall per class: \n    road: %.6f, person: %.6f, car: %.6f, bicycle: %.6f, building: %.6f, wall: %.6f, bridge: %.6f, pole: %.6f, terrain: %.6f, unlabeled: %.6f" \
        % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4],
           recall_per_class[5], recall_per_class[6], recall_per_class[7], recall_per_class[8], recall_per_class[9]))
    print(
        "* iou per class: \n    road: %.6f, person: %.6f, car: %.6f, bicycle: %.6f, building: %.6f, wall: %.6f, bridge: %.6f, pole: %.6f, terrain: %.6f, unlabeled: %.6f" \
        % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4],
           iou_per_class[5], iou_per_class[6], iou_per_class[7], iou_per_class[8], iou_per_class[9]))
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f" \
          % (recall_per_class.mean(), iou_per_class.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f" \
          % (np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class))))
    print(
        '\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' % (
            batch_size, ave_time_cost * 1000 / (len(test_loader) - 5),
            1.0 / (ave_time_cost / (len(test_loader) - 5))))  # ignore the first 10 frames
    print('\n###########################################################################')
