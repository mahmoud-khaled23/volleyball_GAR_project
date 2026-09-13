import yaml

from src.paths import CONFIGS_DIR, DATA_DIR
from src.utils.boxinfo import BoxInfo


data_config_path = CONFIGS_DIR / 'data_config.yaml'
with open(data_config_path, 'r') as file:
    data_config = yaml.load(file, Loader=yaml.FullLoader)

# print(data_config)

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
    train_dirs = data_config['dataset']['train_dirs']
    train_dirs.sort()

    val_dirs = data_config['dataset']['validation_dirs']
    val_dirs.sort()

    test_dirs = data_config['dataset']['test_dirs']
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



# def iterate_videos(videos_lst:list, tracking_annot_path):
#     clips_list_for_loader = []
#     annotations = "annotations.txt"
#     group_activity_encode = prep_categories()[0]
#
#     videos_path = data_config['dataset']['videos']
#
#     for video in videos_lst:
#         if not os.path.exists(os.path.join(videos_path, video)):
#             continue
#
#         video = os.path.join(videos_path, video)
#         clips_dir = os.listdir(os.path.join(videos_path, video))
#         clips_track_annots_dir = os.listdir(os.path.join(tracking_annot_path, video))
#         clips_dir.sort()
#         clips_track_annots_dir.sort()
#
#
#         video_annots = load_video_annots(os.path.join(video, annotations))
#
#         for clip in clips_dir:
#             if not os.path.isdir(os.path.join(video, clip)):
#                 continue
#
#             clip_track_annots_dir = os.listdir(os.path.join(tracking_annot_path, video, clip, f'{clip}.txt'))
#             clip_tracking_annots = load_tracking_annots(clip_track_annots_dir)
#
#             clip_target_frame_annot = video_annots[clip]
#             clip_info = {
#                 'video': video,
#                 'clip': clip,
#                 'tracking_annots': clip_track_annots_dir,
#                 'category': group_activity_encode[clip_target_frame_annot]
#             }
#             clips_list_for_loader.append(clip_info)
#     return clips_list_for_loader


def get_players_boxes(tracking_annot_path):
    # load tracking annotations for one clip
    with open(tracking_annot_path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}

        for line in file:
            box_info = BoxInfo(line)
            player_id, box, lost, category = box_info.player_id, box_info.box, box_info.lost, box_info.category
            box_info_dct = box_info.get_box_info()

            # if number of players is more than 12 by mistake stop on player number 12 and ignore others
            if box_info.player_id > 11:
                continue

            player_boxes[box_info.player_id].append(box_info_dct)

        frame_boxes_dct = {}
        for player_id, boxes_info in player_boxes.items():
            # for baseline 3, I need just 4 frames before and 4 after the target [5:13] from 6 to 14 ignoring zero indexing
            # boxes_info = boxes_info[5:-6]
            if len(boxes_info) == 0:
                continue

            target_boxes_info = boxes_info[9]

            player_box = {
                'player_id': player_id,
                'box': target_boxes_info['box'],
                'category': target_boxes_info['category']
            }

            if target_boxes_info['frame_id'] not in frame_boxes_dct:
                frame_boxes_dct[target_boxes_info['frame_id']] = []

            frame_boxes_dct[target_boxes_info['frame_id']].append(player_box)

        frame_boxes_lst_dct = {}
        for frame_id, boxes_lst in frame_boxes_dct.items():
            player_id = []
            box = []
            category = []

            for boxes in boxes_lst:
                player_id.append(boxes['player_id'])
                box.append(boxes['box'])
                category.append(boxes['category'])

            frame_boxes_lst_dct[frame_id] = [player_id, box, category]
            # print(frame_boxes_lst_dct[frame_id])

        '''
        It returns 9 frames per clip, each frame has 12 player boxes.
        RETURN: --> { 'frame_id': [[player0, player2, ......,player11]
                                   [box0, box1, ......., box11]
                                   [cat0, cat1, ......, cat11]]
                                   }
        '''
        # print(frame_boxes_lst_dct)
        return frame_boxes_lst_dct

def load_tracking_annots(videos_lst:list, tracking_annot_path):
    print(f'Loading tracking annotations for {len(videos_lst)} videos from {tracking_annot_path}')
    clip_track_annots = []
    person_activity_encode = prep_categories()[1]

    for vid in videos_lst:
        if not (tracking_annot_path / vid).exists():
            continue

        # print(vid)
        tracking_annots_clips = sorted((tracking_annot_path / vid).iterdir())
        tracking_annots_clips.sort()
        # print(clips_dir)
        # print(tracking_annots_dir)

        for clip in tracking_annots_clips:
            if not (tracking_annot_path / vid / clip).is_dir():
                continue

            players_boxes = get_players_boxes(tracking_annot_path / vid / clip / f'{clip}.txt')

            for frame_id, boxes in players_boxes.items():
                category = []
                for cat in boxes[2]:
                    category.append(person_activity_encode[cat])

                clip_track_annots_players_dct = {
                    'video': vid,
                    'clip': clip,
                    'frame_id': frame_id,
                    'boxes': boxes[1],
                    'category': category
                }
                # print(clip_track_annots_players_dct)
                clip_track_annots.append(clip_track_annots_players_dct)
            # print(clip_track_annots)
    return clip_track_annots

def prepare_persons_annotations():
    # get train, val and test folders in a sorted way

    annotations_save_path = DATA_DIR / data_config['paths']['persons_annotations']
    if not os.path.exists(annotations_save_path):
        os.makedirs(annotations_save_path)

    tracking_annots_path = DATA_DIR / data_config['paths']['persons_annotations']
    train, val, test = get_videos_dirs()

    train_annots = load_tracking_annots(train, tracking_annots_path)
    print(len(train_annots))
    # train_crops = get_cropped_images(videos_path, train_annots)

    # print('the length of annot : ' + str(len(train_annots)))
    # print(type(train_annots[0]))
    # print((train_annots[0]))
    # print(type(train_annots[0]['box']))

    val_annots = load_tracking_annots(val, tracking_annots_path)
    print(len(val_annots))
    # val_crops = get_cropped_images(videos_path, val_annots)

    test_annots = load_tracking_annots(test, tracking_annots_path)
    # test_crops = get_cropped_images(videos_path, test_annots)

    with open(annotations_save_path / 'train_players_crops.pickle', 'wb') as tr_file:
        pickle.dump(train_annots, tr_file, pickle.HIGHEST_PROTOCOL)

    bytes_size = os.path.getsize(os.path.join(annotations_save_path, 'train_players_crops.pickle'))
    mb_size = bytes_size / (1024 * 1024)

    print(f"File size: {mb_size:.2f} MB")

    with open(annotations_save_path / 'val_players_crops.pickle', 'wb') as vl_file:
        pickle.dump(val_annots, vl_file, pickle.HIGHEST_PROTOCOL)
    #
    with open(annotations_save_path / 'test_players_crops.pickle', 'wb') as ts_file:
        pickle.dump(test_annots, ts_file, pickle.HIGHEST_PROTOCOL)


# def prepare_images_annotations(videos_path, tracking_annot_path):
#     # get train, val and test folders in a sorted way
#     train, val, test = get_videos_dirs(videos_path)
#     image_annotations_to_save = data_config['paths']['image_annotations']
#
#     train_annots = iterate_videos(train, videos_path, tracking_annot_path)
#     val_annots = iterate_videos(val, videos_path, tracking_annot_path)
#     test_annots = iterate_videos(test, videos_path, tracking_annot_path)
#
#     with open(os.path.join(image_annotations_to_save, 'train-target-annot.pickle'), 'wb') as tr_file:
#         pickle.dump(train_annots, tr_file, pickle.HIGHEST_PROTOCOL)
#
#     with open(os.path.join(image_annotations_to_save, 'val-target-annot.pickle'), 'wb') as vl_file:
#         pickle.dump(val_annots, vl_file, pickle.HIGHEST_PROTOCOL)
#
#     with open(os.path.join(image_annotations_to_save, 'test-target-annot.pickle'), 'wb') as ts_file:
#         pickle.dump(test_annots, ts_file, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # root_path, dataset_path, outputs_path = paths(kaggle)
    # print(dataset_path)
    # videos_path = data_config['paths']['videos']
    # output_annot_path, output_training_path = outputs_dirs(outputs_path)
    #
    # image_level = False

    # tracking_annot_path = data_config['paths']['tracking_annotations']
    prepare_persons_annotations()



