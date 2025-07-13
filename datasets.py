from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm


class FaceKeyPointDataset(Dataset):
    def __init__(self, image_path: str, json_path: str, transform=None):
        super().__init__()
        self.images = image_path
        self.annotations = {
            int(k): v
            for k, v in json.load(open(json_path, "r", encoding="utf-8")).items()
        }
        self.transform = transform

    def __getitem__(self, index):
        sample = self.annotations[index]

        landmarks = np.array(sample["face_landmarks"], dtype=np.float32)

        image = cv2.imread(f"{self.images}/{sample['file_name']}")
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(image)

        landmarks[:, :] = np.clip(landmarks[:, :], 0, 511.5)
        landmarks = np.array(
            [x / 2 for x in landmarks[::2]] + [y / 2 for y in landmarks[1::2]]
        )
        image, landmarks = self.apply_transform(image, landmarks)

        landmarks = np.array(
            [x / 256 for x in landmarks[::2]] + [y / 256 for y in landmarks[1::2]]
        )

        return image, landmarks

    def __len__(self):
        return len(self.annotations)

    def apply_transform(self, img, face_landmarks):
        # Применяем трансформацию к изображению
        if self.transform:
            transformed = self.transform(image=img, keypoints=face_landmarks)

            return transformed["image"], transformed["keypoints"]
        else:
            return img, face_landmarks


def show_image_with_landmarks(image, landmarks, title="Image with Landmarks"):
    plt.figure(figsize=(10, 10))
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Теперь форма (h, w, 3)
    plt.imshow(image)

    # Наносим точки
    for x, y in landmarks:
        plt.plot(x, y, "r.", markersize=10)

    plt.title("Image with Landmarks")
    plt.axis("off")
    plt.savefig("output.png")  # Сохраняет изображение в файл
    plt.close()


def create_dataset(images_src, images_dst, annotations, indexes):
    new_annotations = {
        k: v for k, v in annotations.items() if k in indexes
    }  # filter test annotation

    os.makedirs(f"{images_dst}/images", exist_ok=True)

    print(f"create dataset in {images_dst}")

    bar = tqdm(indexes)
    for key in bar:
        shutil.copy(
            f"{images_src}/images/{annotations[key]['file_name']}",
            f"{images_dst}/images/{annotations[key]['file_name']}",
        )

    new_annotations = {
        k: v for k, v in zip(range(len(new_annotations)), new_annotations.values())
    }  # recreate index

    print(f"json saved in {images_dst}/data.json")

    json.dump(new_annotations, open(f"{images_dst}/data.json", "w+"))


def train_test_split(
    original_path, train_path, test_path, original_json: str, test_size
):
    import random

    random.seed(42)
    annotations = {
        int(k): v
        for k, v in json.load(open(original_json, "r", encoding="utf-8")).items()
    }

    test_keys = set(
        random.sample(list(annotations.keys()), int(test_size * len(annotations)))
    )
    train_keys = set(k for k in annotations.keys() if k not in test_keys)

    create_dataset(original_path, train_path, annotations, train_keys)
    create_dataset(original_path, test_path, annotations, test_keys)


if __name__ == "__main__":
    train_test_split("original", "./train", "./test", "./original/all_data.json", 0.2)
