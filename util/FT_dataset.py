import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from PIL import Image


class FT_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=216, input_w=640, transform=[]):
        super(FT_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        self.names = os.listdir(os.path.join(data_dir, split, 'th'))

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

    def read_image_rgb(self, name, folder):
        name = name[:-7] + "0_rgb.png"
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def read_image_label(self, name, folder):
        name = name[:-7] + "0_rgb.npy"
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        label = np.load(file_path)
        return label

    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image_rgb(name, 'rgb')
        image_th = np.expand_dims(self.read_image(name, 'th'), axis=2)
        image = np.concatenate((image, image_th), axis=2)
        label = self.read_image_label(name, 'label')
        for func in self.transform:
            image, label = func(image, label)
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST),
                           dtype=np.int64)
        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data



if __name__ == '__main__':
    # train_dataset = FT_dataset(data_dir="../../../a_dataset/PST900_RGBT_Dataset/PST900_RGBT_Dataset", split='train')
    # val_dataset = FT_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='val')
    test_dataset = FT_dataset(data_dir="../dataset/Freiburg_Thermal_test", split='test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

