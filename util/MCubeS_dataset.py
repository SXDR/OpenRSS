import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from PIL import Image


class MCubeS_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=536, input_w=640, transform=[]):
        super(MCubeS_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, "list_folder", split + '.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def read_image_th(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        # 缩放像素值范围到0到255之间
        min_val = np.min(image)
        max_val = np.max(image)
        scaled_array = ((image - min_val) / (max_val - min_val)) * 255
        # 将浮点数数组转换回整数数组
        scaled_array = scaled_array.astype(np.uint8)
        return scaled_array

    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'polL_color')
        image_th = np.expand_dims(self.read_image_th(name, 'NIR_warped'), axis=2)
        image = np.concatenate((image, image_th), axis=2)
        #label = self.read_image(name, 'SSGT4MS')
        label = self.read_image(name, 'GT')
        for func in self.transform:
            image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST),
                           dtype=np.int64)
        label[label == 255] = 18
        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data


def GT_MCubeS():
    road = [128, 64, 128]
    person = [220, 20, 60]
    car = [0, 0, 142]
    bicycle = [119, 11, 32]
    building = [70, 70, 70]
    wall = [102, 102, 156]
    bridge = [70, 130, 180]
    pole = [153, 153, 153]
    terrain = [152, 251, 152]
    unlabelled = [0, 0, 0]
    palette = np.array(
        [road, person, car, bicycle, building, wall, bridge, pole, terrain, unlabelled])
    return palette


def GT_MCubeS_1():
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


    # ground_sidewalk = [221, 163, 222]
    # curb = [189, 252, 201]
    # fence = [173, 216, 230] wall use
    # vegetation = [255, 204, 204] bridge use
    # sky = [238, 172, 137]

    palette = np.array(
        [road, person, car, bicycle, building, wall, bridge, pole, terrain, unlabelled])
    return palette


def visualize_gt(image_name, predictions, weight_name, data_name):
    palette = eval(data_name)()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)):  # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        # if image_name[i][-4:] == ".png":
        #     image_name[i] = image_name[i][:-4]
        print('../result/Pred/tovs/' + data_name + "/" + weight_name + '_' + image_name[i])
        img.save('../result/Pred/tovs/' + data_name + "/" + weight_name + '_' + image_name[i] + '.png')


if __name__ == '__main__':
    train_dataset = MCubeS_dataset(data_dir="../dataset/multimodal_dataset", split='train')
    val_dataset = MCubeS_dataset(data_dir="../dataset/multimodal_dataset", split='val')
    test_dataset = MCubeS_dataset(data_dir="../dataset/multimodal_dataset", split='test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    for it, (images, labels, names) in enumerate(testloader):
        visualize_gt(image_name=names, predictions=labels, weight_name='Pred_',
                     data_name="GT_MCubeS_1")

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    # print(type(trainloader))
    a, b, c = next(iter(test_dataset))

    # print((a == i))
    print("*" * 50)

    print(a.shape)
    print(b.shape)
    # a = a[0][:3, :, :].permute(1, 2, 0)
    # print(a.shape)

    # print(len(train_dataset))
    # print(type(trainloader))
    # a, b, c = next(iter(trainloader))
    # print(c)
    # print(c[0])
    # print(type(c))
    # print(a.shape)
    # print(a[:, :3].shape)
    # print(b.shape)
    # print(c)
    # integer_list = [1 if 'D' in s else 0.5 for s in c]
    # print(integer_list)

    # a = a*255
    # print(a)
    # print(a.shape)
    # for i in trainloader:
    #     a, b, c = i
    #     print(a)
    #     if 'N' in c[0]:
    #         print("drak")
    #     else:
    #         print(c)
