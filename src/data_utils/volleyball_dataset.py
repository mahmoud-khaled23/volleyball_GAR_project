import os
import pickle
import random

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.paths import VIDEOS_DIR, PERSON_ANNOTATIONS_DIR

from PIL import Image

class VolleyBallPersonDataset(Dataset):

    def __init__(self, data_list, preprocess=None, shuffle=False):
        # super().__init__(self)
        self.data_list = data_list
        self.preprocess = preprocess
        self._shuffle(shuffle)

    # At first, we train on person activity so we will feed the boxes alone with no group activity

    def __getitem__(self, idx):
        """
        Returns a single sample: (processed_crops, labels)

        Returns:
            - processed_crops: tensor [12, 3, 224, 224] (length 12, padded if needed)
            - processed_labels: tensor [12] with -1 for padding
            :param idx:
            :return:
        """
        vid, clip, frame = self.data_list[idx]['video'], self.data_list[idx]['clip'], self.data_list[idx]['frame_id']

        image_path = VIDEOS_DIR / vid / clip / f'{frame}.jpg'
        image = Image.open(image_path).convert('RGB')

        boxes, player_category = self.data_list[idx]['boxes'], self.data_list[idx]['category']

        processed_crops = []
        processed_labels = []

        # ✅ FIXED: Process each box correctly
        for box, label in zip(boxes, player_category):
            # cropped_box = image.crop(box)
            # processed_crop = preprocessor(image.crop(box))
            # processed_crops.append(self.preprocess(image.crop(box)))
            processed_crops.append(self.preprocess(image.crop(box)))
            processed_labels.append(label)

        # Pad to exactly 12 players with zero tensors and -1 labels
        while len(processed_crops) < 12:
            zero_crop = torch.zeros((3, 224, 224), dtype=torch.float32)
            processed_crops.append(zero_crop)
            processed_labels.append(-1)

        # Ensure exactly 12 players (truncate if more)
        processed_crops = torch.stack(processed_crops[:12])
        processed_labels = torch.tensor(processed_labels[:12], dtype=torch.long)

        # ✅ FIXED: Close image to free memory (important with num_workers)
        image.close()

        return processed_crops, processed_labels


    def __len__(self):
        return len(self.data_list)

    def len_12_players(self):
        return len(self.data_list) * 12

    def _shuffle(self, shuffle):
        if shuffle:
            random.shuffle(self.data_list)


def collate_fn(batch):
    crops, labels = zip(*batch)
    images = torch.stack(crops)
    labels = torch.stack(labels)

    return images, labels


from src.utils.preprocessors import person_preprocessor


if __name__ == '__main__':
    print(('-'*20)+' volleyball dataset '+('-'*20))
    # train_dct, val_dct = b1_load()
    person_train_path = PERSON_ANNOTATIONS_DIR / 'train_players_crops.pickle'
    person_val_path = PERSON_ANNOTATIONS_DIR / 'val_players_crops.pickle'

    with open(person_train_path, 'rb') as f:
        person_train = pickle.load(f)

    with open(person_val_path, 'rb') as f:
        person_val = pickle.load(f)

    train_preprocessor, val_preprocessor = person_preprocessor()

    train_data = VolleyBallPersonDataset(person_train, preprocess=train_preprocessor)
    val_data = VolleyBallPersonDataset(person_val, preprocess=val_preprocessor)

    img, cat = train_data[0]
    print(f'image shape: {np.array(img).shape}, category: {cat}')

    cv2.imshow("Image", np.array(img[0].permute(1, 2, 0)))
    # cv2.imshow("Image", np.array(img[0]))
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    # print(f'image : {np.array(img)}, category: {cat}')
    # print(f'image shape: {np.array(img).shape}, category: {cat}')
