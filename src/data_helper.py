import os
import pickle

def get_root_dirs():
    root_path = "/home/ma7moud-5aled/PycharmProjects/volleyball_GAR_project"
    root_dataset_path = os.path.join(root_path, 'volleyball_dataset')
    root_videos_path = os.path.join(root_dataset_path, 'videos')
    root_output_path = os.path.join(root_path, 'outputs')

    return root_path, root_dataset_path, root_videos_path, root_output_path


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


def _get_videos_dirs_lst():
    train_dirs = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31", "32", "36", "38", "39", "40",
                  "41", "42", "48", "50", "52", "53", "54"]
    train_dirs.sort()

    val_dirs = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
    val_dirs.sort()

    test_dirs = ['4', '5', '9', '11', '14', '20', '21', '25', '29', '34', '35', '37', '43', '44', '45', '47']
    test_dirs.sort()

    return train_dirs, val_dirs, test_dirs


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category[clip_dir] = items[1]

        return clip_category


def iterate_videos(root_videos, data_dir):
    clips_list_for_loader = []
    annotations = "annotations.txt"
    encode_cats = prep_categories()[0]

    for data in data_dir:
        if not os.path.exists(os.path.join(root_videos, data)):
            continue

        video = os.path.join(root_videos, data)
        video_clips = os.listdir(os.path.join(root_videos, data))
        video_clips.sort()

        video_annots = load_video_annot(os.path.join(video, annotations))

        for clip in video_clips:
            if not os.path.isdir(os.path.join(video, clip)):
                continue

            clip_target_frame_annot = video_annots[clip]
            clip_info = {
                "video": data,
                "clip": clip,
                "category": encode_cats[clip_target_frame_annot]
            }
            clips_list_for_loader.append(clip_info)
    return clips_list_for_loader


def prepare_dataset(root_videos, output_path):
    # get train, val and test folders in a sorted way
    train, val, test = get_videos_dirs_lst()

    train_annots = iterate_videos(root_videos, train)
    val_annots = iterate_videos(root_videos, val)
    test_annots = iterate_videos(root_videos, test)

    with open(os.path.join(output_path, "train-target-annot.pickle"), 'wb') as tr_file:
        pickle.dump(train_annots, tr_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, "val-target-annot.pickle"), 'wb') as vl_file:
        pickle.dump(val_annots, vl_file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, "test-target-annot.pickle"), 'wb') as ts_file:
        pickle.dump(test_annots, ts_file, pickle.HIGHEST_PROTOCOL)

def b1_load(root_output, test=False):
    train_pkl = root_output + "/structured-data/volleyball-annotations/train-target-annot.pickle"
    val_pkl = root_output + "/structured-data/volleyball-annotations/val-target-annot.pickle"
    test_pkl = root_output + "/structured-data/volleyball-annotations/test-target-annot.pickle"

    if not test:
        with open(train_pkl, 'rb') as tr:
            train_dct = pickle.load(tr)

        with open(val_pkl, 'rb') as vl:
            val_dct = pickle.load(vl)

        return train_dct, val_dct

    else:
        with open(test_pkl, 'rb') as ts:
            test_dct = pickle.load(ts)

        return test_dct


if __name__ == '__main__':
    root_videos_path = '/volleyball_dataset/videos'
    # prepare_testset()

