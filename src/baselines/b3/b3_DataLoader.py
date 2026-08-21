import os
import random

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

class VolleyBallPersonDataLevel(Dataset):

    def __init__(self, videos_path, data_list, preprocess=None, shuffle=False):
        # super().__init__(self)
        self.videos_path = videos_path
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
        """
        vid, clip, frame = self.data_list[idx]['video'], self.data_list[idx]['clip'], self.data_list[idx]['frame_id']

        image_path = os.path.join(self.videos_path, vid, clip, f'{frame}.jpg')
        image = Image.open(image_path).convert('RGB')

        boxes, player_category = self.data_list[idx]['boxes'], self.data_list[idx]['category']

        processed_crops = []
        processed_labels = []

        # ✅ FIXED: Process each box correctly
        for box, label in zip(boxes, player_category):
            # cropped_box = image.crop(box)
            # processed_crop = preprocessor(image.crop(box))
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
        return len(self.data_list) * len(self.data_list[0]['boxes'])

    def _shuffle(self, shuffle):
        if shuffle:
            random.shuffle(self.data_list)


def collate_fn(batch):
    # batch is a list of samples
    # each sample looks like:
    # [player_ids, player_crops, player_categories]

    max_players = 12

    batch_crops = []
    # batch_labels = []
    # for sample in batch:
    #     frame_images = []
    #     frame_labels = []
    #
    #     player_ids, player_crops, player_categories = sample
    #
    #     # each of these is already a list of length 12
    #     for crop, label in zip(player_crops, player_categories):
    #         crop = preprocessor(crop)
    #
    #         frame_images.append(crop)
    #         frame_labels.append(label)
    #
    #     batch_crops.append(torch.stack(frame_images))
    #     batch_labels.append(torch.tensor(frame_labels, dtype=torch.long))
    #
    # # convert to tensors
    # images = torch.stack(batch_crops)  # [B, 12, 3, 224, 224]
    # labels = torch.stack(batch_labels)  # [B, 12]

    crops, labels = zip(*batch)
    images = torch.stack(crops)
    labels = torch.stack(labels)

    return images, labels






        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # def preprocessors(image_level):
    #     if image_level:
    #         train_preprocess = transforms.Compose([
    #             transforms.Resize((256, 256)),
    #             transforms.CenterCrop((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         ])
    #
    #         test_preprocess = transforms.Compose([
    #             transforms.Resize((256, 256)),
    #             transforms.CenterCrop((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         ])
    #     else:
    #             train_preprocess = transforms.Compose([
    #                 transforms.Resize((224, 224)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #             ])
    #
    #             test_preprocess = transforms.Compose([
    #                 transforms.Resize((256, 256)),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #             ])
    #
    #     return train_preprocess, test_preprocess

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
