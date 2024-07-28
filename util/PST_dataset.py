import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from PIL import Image


class PST_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=360, input_w=640, transform=[]):
        super(PST_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        self.names = os.listdir(os.path.join(data_dir, split, 'rgb'))

        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'rgb')
        image_th = np.expand_dims(self.read_image(name, 'thermal'), axis=2)
        image = np.concatenate((image, image_th), axis=2)
        label = self.read_image(name, 'labels')
        for func in self.transform:
            image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST),
                           dtype=np.int64)
        return torch.tensor(image), torch.tensor(label), name, name

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    train_dataset = PST_dataset(data_dir="../../../a_dataset/PST900_RGBT_Dataset/PST900_RGBT_Dataset", split='train')
    # val_dataset = PST_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='val')
    test_dataset = PST_dataset(data_dir="../../../a_dataset/PST900_RGBT_Dataset/PST900_RGBT_Dataset", split='test')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

    # print(len(train_dataset))
    # print(type(trainloader))
    a, b, c = next(iter(trainloader))

    # print((a == i))
    print("*" * 50)

    print(a.shape)
    print(b.shape)

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
