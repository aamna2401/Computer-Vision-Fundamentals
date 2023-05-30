import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np


def image_paths(path):
    celebs = os.listdir(path)
    celebs_imgs = {}
    max_images = 0
    for c in tqdm(celebs):
        img_list = os.listdir(os.path.join(path, c))
        celebs_imgs[c] = [os.path.join(path, c, i) for i in img_list]
        if len(img_list) > max_images:
            max_images = len(img_list)
    return celebs_imgs, max_images


def get_transforms():
    threshold = np.random.randint(0, 255)
    trans = [
        transforms.ColorJitter(brightness=np.random.rand(),
                               contrast=np.random.rand(),
                               saturation=np.random.rand(),
                               hue=np.random.rand()),
        transforms.Grayscale(3),
        transforms.Pad(np.random.randint(1, 5)),
        transforms.RandomAffine(degrees=np.random.rand() * 360),
        transforms.RandomCrop(np.random.randint(100, 150)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomPerspective(np.random.rand(), p=1),
        transforms.RandomRotation(degrees=np.random.rand() * 360),
        transforms.RandomVerticalFlip(p=1),
        transforms.GaussianBlur(3),
        transforms.RandomInvert(p=1),
        transforms.RandomPosterize(np.random.randint(1, 8), p=1),
        transforms.RandomSolarize(threshold, p=1),
        transforms.RandomAdjustSharpness(np.random.randint(0, 3), p=1),
        transforms.RandomAutocontrast(p=1),
        transforms.RandomEqualize(p=1),
        transforms.RandomResizedCrop(256),
    ]
    return trans


def apply_transforms(path):
    print('Reading images relative path...')
    celebs, max_images = image_paths(path)
    trans = get_transforms()
    print('Transforming images...')
    for celeb, images in tqdm(celebs.items()):
        if len(images) < max_images:
            print(len(images))
            for i in range(max_images - len(images) + 1):
                img_path = images[np.random.randint(0, len(images))]
                transform = trans[np.random.randint(0, len(trans))]
                # applying transform
                img = transform.forward(Image.open(img_path))
                # saving image
                split = img_path.split('.')
                num = str(np.random.randint(0, 1000))
                new_path = split[0] + '_' + num + '.' + split[1]
                print(new_path)
                img.save(new_path)


if __name__ == "__main__":
    dataset_dir = 'root'
    apply_transforms(dataset_dir)

