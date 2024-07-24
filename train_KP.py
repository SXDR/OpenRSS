import sys
import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import random
import warnings
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.KP_dataset import KP_dataset
from util.augmentation import RandomFlip, RandomCrop
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from pytorch_toolbelt import losses as L
from loss_hub.losses import DiceLoss, SoftCrossEntropyLoss

from torch.cuda.amp import autocast, GradScaler
from semseg.models import *
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from segment_anything import sam_model_registry
import sam_lora_image_encoder
from segment_anything.utils import transforms
from segment_anything.utils.amg import build_all_layer_point_grids
import clip


#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='openrss')
parser.add_argument('--batch_size', '-b', type=int, default=8)
parser.add_argument('--channel', '-c', type=int, default=3)
parser.add_argument('--lr_start', '-ls', type=float, default=0.0005)
parser.add_argument('--seed', '-seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=300)  # please stop training mannully
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--num_classes', '-ncs', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=20)
parser.add_argument('--accumulation_steps', '-as', type=int, default=4)
parser.add_argument('--img_size', type=int,
                    default=640, help='input patch size of network input')
parser.add_argument('--vit_name', type=str,
                    default='vit_h', help='select one vit model')
parser.add_argument('--cm_type', type=str,
                    default='jet', help='select cm_type')
parser.add_argument('--ckpt', type=str, default='/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--data_dir', '-dr', type=str, default='../KP_dataset')
parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")
parser.add_argument('--resume', default='', help='resume from checkpoint')

args = parser.parse_args()
#############################################################################################
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
]


def train(epo, model, text_features, points, train_loader, optimizer, scheduler, scaler=None, multimask_output=True):
    model.train()
    for it, (images, labels, names, th) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        start_t = time.time()  # time.time() returns the current time
        DiceLoss_fn = DiceLoss(mode="multiclass")
        SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
        criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.5,
                                second_weight=0.5).cuda()
        with autocast(enabled=scaler is not None):
            logits = model(images, points, text_features, multimask_output)
            loss = criterion(logits["masks"], labels)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True  # note that I have not colorized the GT and predictions here
        if accIter['train'] % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                    padding=10)  # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1,
                            255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),
                                               1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits["masks"].argmax(1).unsqueeze(
                    1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),
                                             1)  # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1
        #torch.cuda.empty_cache()


def validation(epo, model, text_features, points, val_loader, multimask_output):
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names, th) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time()  # time.time() returns the current time
            logits = model(images, points, text_features, multimask_output)
            loss = F.cross_entropy(logits["masks"], labels)
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (
                      args.model_name, epo, args.epoch_max, it + 1, len(val_loader),
                      len(names) / (time.time() - start_t), float(loss),
                      datetime.datetime.now().replace(microsecond=0) - start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                        padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1,
                                255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),
                                                   1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits["masks"].argmax(1).unsqueeze(
                        1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),
                                                 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1


def testing(epo, model, text_features, points, test_loader, multimask_output):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["road", "sidewalk", "building", "wall", "fence", "pole",
                 "traffic_light", "traffic_sign", "vegetation", "terrain",
                 "sky", "person", "rider", "car", "truck",
                 "bus", "train", "motorcycle", "bicycle", "Background"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names, th) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images, points, text_features, multimask_output)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits["masks"].argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
                args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
                datetime.datetime.now().replace(microsecond=0) - start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
                args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)


if __name__ == '__main__':
    class_num = ["road", "sidewalk", "building", "wall", "fence", "pole",
                 "traffic_light", "traffic_sign", "vegetation", "terrain",
                 "sky", "person", "rider", "car", "truck",
                 "bus", "train", "motorcycle", "bicycle", "Background"]
    text_list = []
    qz = "a pohot of "
    for i in class_num:
        text_list.append(qz+i)
    device_clip = "cpu"
    model_clip, preprocess = clip.load("/ViT-L-14.pt", device=device_clip)
    text = clip.tokenize(text_list).to(device_clip)
    with torch.no_grad():
        text_features = model_clip.encode_text(text)
    text_features = text_features.cuda(args.gpu)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    train_dataset = KP_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset = KP_dataset(data_dir=args.data_dir, split='val')
    test_dataset = KP_dataset(data_dir=args.data_dir, split='test')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size= args.img_size, num_classes=args.num_classes,
                                                                checkpoint=args.ckpt)

    model = sam_lora_image_encoder.LoRA_Sam(sam, args.rank)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    
    
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
    
    #iters_per_epoch = len(train_dataset) // (args.batch_size * args.accumulation_steps)
    iters_per_epoch = len(train_dataset) // (args.batch_size)
    optimizer = get_optimizer(model, "adamw", args.lr_start, 0.01)
    scheduler = get_scheduler("warmuppolylr", optimizer, args.epoch_max * iters_per_epoch, 7, iters_per_epoch * 10,
                              0.1)
    scaler = GradScaler() if args.amp else None
    if args.lora_ckpt is not None:
        model.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    weight_dir = os.path.join("runs_openrss/", args.model_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    os.chmod(weight_dir,
             stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    writer = SummaryWriter("./runs_openrss/tensorboard_log")
    os.chmod("runs_openrss/tensorboard_log",
             stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("runs_openrss", stat.S_IRWXO)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    config_file = os.path.join(weight_dir, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        train(epo, model, text_features, points, train_loader, optimizer, scheduler, scaler, multimask_output)
        val_loss = validation(epo, model, text_features, points, val_loader, multimask_output)
        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)

        try:
            model.save_lora_parameters(checkpoint_model_file)
        except:
            model.module.save_lora_parameters(checkpoint_model_file)

        testing(epo, model, text_features, points, test_loader, multimask_output)
