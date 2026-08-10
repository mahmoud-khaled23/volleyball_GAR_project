# import numpy as np # linear algebra
# import pandas as pd # data processing

import pathlib

import os


def paths(is_kaggle=False):
    root_path = pathlib.Path.cwd().parents[0]
    print(root_path)

    if is_kaggle:
        output_path = os.path.join(root_path, 'working')
        dataset_path = os.path.join(root_path, 'input/volleyball')

    else:
        output_path = os.path.join(root_path, 'outputs')
        dataset_path = os.path.join(root_path, 'volleyball')


    # output_dir = os.path.join(root_path, "outputs")
    return root_path, dataset_path, output_path

def outputs_dirs(outputs_path):

    output_annot_path = os.path.join(outputs_path, 'baselines-annotations')

    # The if condition is for Kaggle because of the every restart of the session
    if not os.path.exists(output_annot_path):
        os.makedirs(output_annot_path)

    output_training_path = os.path.join(outputs_path, 'training-outputs')

    # The if condition is for Kaggle because of the every restart of the session
    if not os.path.exists(output_training_path):
        os.makedirs(output_training_path)

    return  output_annot_path, output_training_path

def videos_path(dataset_path):
    return os.path.join(dataset_path, 'volleyball_/videos')


# print(paths(is_kaggle=True))
print(paths(is_kaggle=False))


import os
import pickle

def prep_categories():
    group_categories = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

    person_categories = {
        'standing': 0,
        'setting': 1,
        'waiting': 2,
        'moving': 3,
        'falling': 4,
        'spiking': 5,
        'jumping': 6,
        'digging': 7,
        'blocking': 8
    }

    return group_categories, person_categories


def get_videos_dirs(path='./'):
    train_dirs = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40",
                  "41", "42", "48", "50", "52", "53", "54"]
    train_dirs.sort()

    val_dirs = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
    val_dirs.sort()

    test_dirs = ['4', '5', '9', '11', '14', '20', '21', '25', '29', '34', '35', '37', '43', '44', '45', '47']
    test_dirs.sort()

    return train_dirs, val_dirs, test_dirs

# for >>> baseline 1 <<<, we get the target frame from the single clip at the video
# the dataset is formated as videos/annotations.txt
# the text file annotations.txt is formated as at the first column 'target_frame_no of the single clip of the video then group activity then single annot of every single player with bounding box'
# the function below get [clip_no, group activity]
def load_video_annots(video_annot):
    with open(video_annot, 'r') as file:
        clip_category = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category[clip_dir] = items[1]

        return clip_category

from boxinfo import BoxInfo

tracking_annot_path = 'volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation'
def load_tracking_annots(tracking_annot_path):
    # load tracking annotations for one clip
    with open(tracking_annot_path, 'r') as file:
        player_boxes = {idx:[] for idx in range(12)}
        frame_boxes_dct = {}

        for line in file:
            box_info = BoxInfo(line)
            # if number of players is more than 12 by mistake stop on player number 12 and ignore others
            if box_info.player_id > 11:
                continue
            player_boxes[box_info.player_id].append(box_info)

        for player_id, boxes_info in player_boxes.items():
            # for baseline 3, I need just 4 frames before and 4 after the target [5:13] from 6 to 14 ignoring zero indexing
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_id not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_id] = []

                frame_boxes_dct[box_info.frame_id].append(box_info)

        return frame_boxes_dct



def iterate_videos(videos_dir, videos_path, tracking_annot_path):
    clips_list_for_loader = []
    annotations = "annotations.txt"
    group_activity_encode = prep_categories()[0]

    for video in videos_dir:
        if not os.path.exists(os.path.join(videos_path, video)):
            continue

        video = os.path.join(videos_path, video)
        clips_dir = os.listdir(os.path.join(videos_path, video))
        clips_track_annots_dir = os.listdir(os.path.join(tracking_annot_path, video))
        clips_dir.sort()
        clips_track_annots_dir.sort()


        video_annots = load_video_annots(os.path.join(video, annotations))

        for clip in clips_dir:
            if not os.path.isdir(os.path.join(video, clip)):
                continue

            clip_track_annots_dir = os.listdir(os.path.join(tracking_annot_path, video, clip, f'{clip}.txt'))
            clip_tracking_annots = load_tracking_annots(clip_track_annots_dir)

            clip_target_frame_annot = video_annots[clip]
            clip_info = {
                'video': video,
                'clip': clip,
                'tracking_annots': clip_track_annots_dir,
                'category': group_activity_encode[clip_target_frame_annot]
            }
            clips_list_for_loader.append(clip_info)
    return clips_list_for_loader


def prepare_dataset(videos_path, tracking_annot_path):
    # get train, val and test folders in a sorted way
    train, val, test = get_videos_dirs(videos_path)

    train_annots = iterate_videos(train, videos_path, tracking_annot_path)
    val_annots = iterate_videos(val, videos_path, tracking_annot_path)
    test_annots = iterate_videos(test, videos_path, tracking_annot_path)

    with open(os.path.join(output_annot_path, 'train-target-annot.pickle'), 'wb') as tr_file:
        pickle.dump(train_annots, tr_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_annot_path, 'val-target-annot.pickle'), 'wb') as vl_file:
        pickle.dump(val_annots, vl_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_annot_path, 'test-target-annot.pickle'), 'wb') as ts_file:
        pickle.dump(test_annots, ts_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    kaggle = False
    root_path, dataset_path, outputs_path = paths(kaggle)
    print(dataset_path)
    videos_path = videos_path(root_path)
    output_annot_path, output_training_path = outputs_dirs(outputs_path)

    image_level = False

    tracking_annot_path = os.path.join(dataset_path, 'volleyball_tracking_annot')
    prepare_dataset(videos_path)



