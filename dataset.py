from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os


class ExpWDataset(Dataset):
    def __init__(self, data_root, index_root, input_size, augment):
        self.data = []
        self.data_root = data_root
        self.input_size = input_size
        self.augment = augment

        with open(index_root, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                content = line.split(' ')
                path = os.path.join(self.data_root, content[0])
                if os.path.exists(path) and int(content[7]) != 6:
                    self.data.append([path, int(content[3]), int(content[2]), int(content[4]), int(content[5]), int(content[7])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.resize(cv2.imread(self.data[idx][0], 0)[self.data[idx][2]:self.data[idx][4], self.data[idx][1]:self.data[idx][3]], (self.input_size, self.input_size))
        image = preprocess(image, self.input_size, self.augment)
        label = self.data[idx][5]
        return image, label


def preprocess(image, input_size, augmentation=True):
    if augmentation:
        crop_transform = transforms.Compose([
            transforms.Resize(input_size // 4 * 5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(input_size),
            transforms.RandomRotation(10)])
    else:
        crop_transform = transforms.CenterCrop(input_size)

    result = transforms.Compose([
        transforms.ToPILImage(),
        crop_transform,
        transforms.ToTensor(),
    ])(image)
    return result
