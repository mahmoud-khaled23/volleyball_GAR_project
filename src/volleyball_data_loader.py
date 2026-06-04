import os
import random


import torchvision.transforms as transforms

import torch
from torch.utils.data import Dataset
from PIL import Image

class VolleyBallDataSet(Dataset):

    def __init__(self, root_videos_path, data_list, preprocess=None, shuffle=False):
        # super().__init__(self)
        self.root_videos_path = root_videos_path
        self.data_list = data_list
        self.preprocess = preprocess
        self._shuffle(shuffle)


    def __getitem__(self, idx):
        video = self.data_list[idx]["video"]
        clip = self.data_list[idx]["clip"]
        category =  self.data_list[idx]["category"]

        image_path = os.path.join(self.root_videos_path, video, clip, f'{clip}.jpg')
        image = Image.open(image_path).convert('RGB')

        if self.preprocess:
            image = self.preprocess(image)

        category = torch.tensor(category)

        return image, category

    def __len__(self):
        return len(self.data_list)

    def _shuffle(self, shuffle):
        if shuffle:
            random.shuffle(self.data_list)


def get_preprocess():
    train_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_preprocess, test_preprocess

#
# if __name__ == '__main__':
#     root, root_dataset, root_videos, root_output = get_root_dirs()
#
#     train_dct, val_dct = b1_load()
#     preprocess = get_preprocess()
#
#     train_data = VolleyBallDataSet(root_videos, train_dct, preprocess=preprocess)
#     val_data = VolleyBallDataSet(root_videos, val_dct, preprocess=preprocess)
#
#     img, cat = train_data[0]
#     print(f'image shape: {np.array(img).shape}, category: {cat}')
#
#     # cv2.imshow("Image", np.array(img.permute(1, 2, 0)))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # print(f'image : {np.array(img)}, category: {cat}')
#     print(f'image shape: {np.array(img).shape}, category: {cat}')
