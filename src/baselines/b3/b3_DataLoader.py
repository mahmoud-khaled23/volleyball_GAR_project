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
        vid, clip, frame = self.data_list[idx]['vid'], self.data_list[idx]['clip'], self.data_list[idx]['frame']

        image_path = os.path.join(self.videos_path, vid, clip, f'{frame}.jpg')
        image = Image.open(image_path).convert('RGB')

        boxes, player_category = self.data_list[idx]['boxes'], self.data_list[idx]['category']

        processed_crops = []
        processed_labels = []

        # ✅ FIXED: Process each box correctly
        for box, label in zip(boxes, player_category):
            # cropped_box = image.crop(box)
            # processed_crop = preprocessor(image.crop(box))
            processed_crops.append(preprocessor(image.crop(box)))
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


    # def collate_fn(self, batch):
    #     frames = {}
    #     video, clip = None, None
    #     for item in batch:
    #         video, clip = item["video"], item["clip"]
    #         frame_key = (item["video"], item["clip"], item["frame_id"])
    #         frames.setdefault(frame_key, []).append(item)
    #
    #     batch_size = len(frames)
    #     images = torch.zeros((batch_size, 12, 3, 224, 224), dtype=torch.float32)
    #     targets = torch.full((batch_size, 12), -1, dtype=torch.long)
    #     # mask = torch.zeros((batch_size, 12), dtype=torch.bool)
    #
    #     for frame_index, items in enumerate(frames.values()):
    #         image_path = os.path.join(self.root_videos_path, frame_index[0], frame_index[1], f'{frame_index[2]}.jpg')
    #         image = Image.open(image_path).convert('RGB')
    #
    #         for item in items:
    #             pid = item["player_id"]
    #             if pid < 0 or pid >= 12:
    #                 continue
    #
    #             x1, y1, x2, y2 = item["frame_id"]['box']
    #             cropped_image = image.crop((x1, y1, x2, y2))
    #             if self.preprocess:
    #                 cropped_image = _preprocessor(cropped_image)
    #
    #             category = item['frame_id']["category"]
    #             images[frame_index, pid] = item[frame_index]["image"]
    #             targets[frame_index, pid] = item["category"]
    #             # mask[frame_index, pid] = True
    #
    #     return images, targets # mask

    def collate_fn_zero_fill(self, batch):
        """
        Fill missing players with zero vectors.
        Assumes batch items are tuples of (player_sequence, label) or similar.
        """
        player_id = []
        sequences = []
        labels = []

        for item in batch:
            # item could be: (sequence_dict, label) or (player_list, label)

            player_seq, label = item

            # Ensure we have exactly 12 players
            filled_seq = self.fill_missing_players_with_zeros(player_seq, num_players=12)

            sequences.append(filled_seq)
            labels.append(label)

        # Stack into tensors
        sequences = torch.stack(sequences)
        labels = torch.tensor(labels)

        return sequences, labels

    def fill_missing_players_with_zeros(self, player_seq, num_players=12):
        """
        Fill gaps in player sequence with zero vectors.
        Assumes player_seq is a dict with player_ids and features.
        """
        # If using dict format: {0: features, 2: features, ...} (missing player 1)
        if isinstance(player_seq, dict):
            feature_dim = list(player_seq.values())[0].shape[0] if player_seq else 128
            filled = torch.zeros(num_players, feature_dim)

            for pos, features in player_seq.items():
                filled[pos] = features

            return filled

        return player_seq

    def __len__(self):
        return len(self.data_list)

    def _shuffle(self, shuffle):
        if shuffle:
            random.shuffle(self.data_list)

def preprocessor(image):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image)

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
