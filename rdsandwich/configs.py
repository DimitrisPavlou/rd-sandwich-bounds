"""
Small registries shared across scripts, mirroring ``configs.py``.
"""

# ImageNet class names used in the GAN-generated-image experiments (Sec. 6.3);
# extend as needed. Keys are the `--dataset` CLI values, values are the class
# names understood by pytorch-pretrained-biggan's one_hot_from_names.
biggan_class_names_to_ids = {
    "basenji": "basenji",
}

KODAK_URL = "http://r0k.us/graphics/kodak"
TECNICK_URL = "https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
