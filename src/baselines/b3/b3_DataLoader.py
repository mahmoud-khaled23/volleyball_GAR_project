import os
import random

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

class VolleyBallPersonDataLevel(Dataset):

    def __init__(self, root_videos_path, data_list, preprocess=None, shuffle=False):
        # super().__init__(self)
        self.root_videos_path = root_videos_path
        self.data_list = data_list
        self.preprocess = preprocess
        self._shuffle(shuffle)

    # At first, we train on person activity so we will feed the boxes alone with no group activity

    def __getitem__(self, idx):
        video = self.data_list[idx]["video"]
        clip = self.data_list[idx]["clip"]

        player_id = self.data_list[idx]["player_id"]
        frame_id = self.data_list[idx]["frame_id"]
        box =  self.data_list[idx]['box']
        lost = self.data_list[idx]['lost']
        category =  self.data_list[idx]["category"]

        image_path = os.path.join(self.root_videos_path, video, clip, f'{frame_id}.jpg')
        image = Image.open(image_path).convert('RGB')

        x1, y1, x2, y2 = box
        cropped_image = image.crop((x1, y1, x2, y2))
        
        if self.preprocess:
            cropped_image = _preprocess(cropped_image)

        category = torch.tensor(category)

        return cropped_image, category

    def __len__(self):
        return len(self.data_list)

    def _shuffle(self, shuffle):
        if shuffle:
            random.shuffle(self.data_list)

def _preprocess(image):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def preprocessors(image_level):
    if image_level:
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
    else:
            train_preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
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
