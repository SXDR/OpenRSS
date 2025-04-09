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





if __name__ == '__main__':
    train_dataset = MCubeS_dataset(data_dir="../dataset/multimodal_dataset", split='train')
    val_dataset = MCubeS_dataset(data_dir="../dataset/multimodal_dataset", split='val')
    test_dataset = MCubeS_dataset(data_dir="../dataset/multimodal_dataset", split='test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)




