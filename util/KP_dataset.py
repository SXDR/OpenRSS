import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from PIL import Image


class KP_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=512, input_w=640, transform=[]):
        super(KP_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder):
        splited_name = name.split('_')
        file_path = os.path.join(self.data_dir, 'images', splited_name[0], splited_name[1], folder,
                                 splited_name[2].replace('png', 'jpg', 1))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def read_label(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'visible')
        image_th = np.uint8(np.expand_dims(self.read_image(name, 'lwir').mean(axis=2), axis=2))
        image = np.concatenate((image, image_th), axis=2)
        label = self.read_label(name, 'labels')
        for func in self.transform:
            image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST),
                           dtype=np.int64)
        return torch.tensor(image),  torch.tensor(label), name, name

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    train_dataset = KP_dataset(data_dir="../../../a_dataset/KPdataset_concise/KPdataset_concise", split='train')
    # val_dataset = KP_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='val')
    # test_dataset = KP_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='test')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

