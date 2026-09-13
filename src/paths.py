import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / 'configs'

DATA_DIR = PROJECT_ROOT / 'data'
VIDEOS_DIR = DATA_DIR / 'volleyball' / 'volleyball_' / 'videos'
DEFAULT_ANNOTATIONS_DIR = DATA_DIR / 'volleyball' / 'volleyball_tracking_annotation' /'volleyball_tracking_annotation' /'annotations'

PERSON_ANNOTATIONS_DIR = DATA_DIR / 'persons_annotations'
IMAGE_ANNOTATIONS_DIR = DATA_DIR / 'images_annotations'

OUTPUT_DIR = PROJECT_ROOT / 'outputs'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# print(PROJECT_ROOT)
# print(CONFIGS_DIR)
# print(DATA_DIR)
# print(OUTPUT_DIR)

