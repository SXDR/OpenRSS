import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from PIL import Image
from matplotlib import cm


class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[], cm_type="jet"):
        super(MF_dataset, self).__init__()

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
        self.cm_type = cm_type

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')
        for func in self.transform:
            image, label = func(image, label)
        th = image[:, :, 3]
        cmap = cm.get_cmap(self.cm_type)
        th = (cmap(th)[:, :, :3] * 255).astype(np.uint8)
        th = np.asarray(PIL.Image.fromarray(th).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST),
                           dtype=np.int64)
        return torch.tensor(image), torch.tensor(label), name, th

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    train_dataset = MF_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='train')
    val_dataset = MF_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='val')
    test_dataset = MF_dataset(data_dir="../../../a_dataset/MFNet_dataset", split='test')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

