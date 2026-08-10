# import numpy as np # linear algebra
# import pandas as pd # data processing

import pathlib

import os

#
# def paths(is_kaggle=False):
#     root_path = pathlib.Path.cwd().parents[0]
#     print(root_path)
#
#     if is_kaggle:
#         output_path = os.path.join(root_path, 'working')
#         dataset_path = os.path.join(root_path, 'input/volleyball')
#
#     else:
#         output_path = os.path.join(root_path, 'outputs')
#         dataset_path = os.path.join(root_path, 'volleyball')
#
#
#     # output_dir = os.path.join(root_path, "outputs")
#     return root_path, dataset_path, output_path

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
# print(paths(is_kaggle=False))


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


def get_videos_dirs():
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

from src.boxinfo import BoxInfo

# tracking_annot_path = 'volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation'
def get_players_boxes(tracking_annot_path):
    # load tracking annotations for one clip
    with open(tracking_annot_path, 'r') as file:
        player_boxes = {idx:[] for idx in range(12)}

        for line in file:
            box_info = BoxInfo(line)
            player_id, box, lost, category = box_info.player_id, box_info.box, box_info.lost, box_info.category
            box_info_dct = box_info.get_box_info()

            # if number of players is more than 12 by mistake stop on player number 12 and ignore others
            if box_info.player_id > 11:
                continue

            player_boxes[box_info.player_id].append(box_info_dct)

        for player_id, boxes_info in player_boxes.items():
            # for baseline 3, I need just 4 frames before and 4 after the target [5:13] from 6 to 14 ignoring zero indexing
            player_boxes[player_id] = boxes_info[5:]
            player_boxes[player_id] = boxes_info[:-6]
        return player_boxes

def load_tracking_annots(dirs_list, tracking_annot_path: str):
    clip_track_annots = []
    person_activity_encode = prep_categories()[1]

    for vid in dirs_list:
        if not os.path.exists(os.path.join(tracking_annot_path, vid)):
            continue

        tracking_annots_vid_dir = os.listdir(os.path.join(tracking_annot_path, vid))
        tracking_annots_vid_dir.sort()
        # print(clips_dir)
        # print(tracking_annots_dir)

        for clip in tracking_annots_vid_dir:
            if not os.path.isdir(os.path.join(tracking_annot_path, vid, clip)):
                continue

            players_boxes = get_players_boxes(os.path.join(tracking_annot_path, vid, clip, f'{clip}.txt'))
            # print(f'{vid} : {players_boxes}')
            for player_id, boxes_info in players_boxes.items():

                for box in boxes_info:
                    clip_track_annots_players_dct = {
                        'video': vid,
                        'clip': clip,
                        'player_id': player_id,
                        'frame_id': box['frame_id'],
                        'box': box['box'],
                        'lost': box['lost'],
                        'category': person_activity_encode[box['category']]
                    }
                    clip_track_annots.append(clip_track_annots_players_dct)
        # print(len(clip_track_annots))
    return clip_track_annots


def prepare_dataset():
    # get train, val and test folders in a sorted way

    root_path = pathlib.Path.cwd().parents[2]
    output_annot_path = os.path.join(root_path, 'outputs', 'b3_data_structure', 'annots')
    if not os.path.exists(output_annot_path):
        os.makedirs(output_annot_path)

    # videos_path = os.path.join(root_path, 'volleyball/volleyball_/videos')
    tracking_annot_path = os.path.join(root_path,
                                            'volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation')
    print(root_path)
    train, val, test = get_videos_dirs()

    train_annots = load_tracking_annots(train, tracking_annot_path)
    print('the length of annot : ' + str(len(train_annots)))
    print(type(train_annots[0]))
    print((train_annots[0]))
    print(type(train_annots[0]['box']))

    val_annots = load_tracking_annots(val, tracking_annot_path)
    test_annots = load_tracking_annots(test, tracking_annot_path)

    with open(os.path.join(output_annot_path, 'train-target-annot.pickle'), 'wb') as tr_file:
        pickle.dump(train_annots, tr_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_annot_path, 'val-target-annot.pickle'), 'wb') as vl_file:
        pickle.dump(val_annots, vl_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_annot_path, 'test-target-annot.pickle'), 'wb') as ts_file:
        pickle.dump(test_annots, ts_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    kaggle = False
    # root_path, dataset_path, outputs_path = paths(kaggle)
    # print(dataset_path)
    # videos_path = videos_path(root_path)
    # output_annot_path, output_training_path = outputs_dirs(outputs_path)
    #
    # image_level = False
    #
    # tracking_annot_path = os.path.join(dataset_path, 'volleyball_tracking_annot')
    prepare_dataset()



