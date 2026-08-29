import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomBrightnessContrast,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotate90,
    RandomVerticalFlip,
    RandomTemporalSwap,
)


class LEVIRCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        transform=None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.root = root
        self.split = split
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.augment = bool(augment)
        self.normalize = bool(normalize)

        if transform is None:
            transforms = []
            if self.augment:
                transforms.extend(
                    [
                        RandomResizedCrop(),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        RandomRotate90(),
                        RandomTemporalSwap(),
                        RandomBrightnessContrast(),
                        ColorJitter(),
                        GaussianBlur(),
                    ]
                )
            if self.normalize:
                transforms.append(Normalize(mean=mean, std=std))
            self.transform = Compose(transforms) if transforms else None
        else:
            self.transform = transform

        split_pre_dir = os.path.join(root, split, "A")
        split_post_dir = os.path.join(root, split, "B")
        split_label_dir = os.path.join(root, split, "label")

        flat_pre_dir = os.path.join(root, "A")
        flat_post_dir = os.path.join(root, "B")
        flat_label_dir = os.path.join(root, "label")

        if os.path.isdir(split_pre_dir):
            pre_dir, post_dir, label_dir = split_pre_dir, split_post_dir, split_label_dir
            self.pre_files = self._list_images(pre_dir)
            self.post_files = self._list_images(post_dir)
            self.label_files = self._list_images(label_dir)
        elif os.path.isdir(flat_pre_dir):
            pre_dir, post_dir, label_dir = flat_pre_dir, flat_post_dir, flat_label_dir
            list_dir = os.path.join(root, "list")
            list_file = self._find_list_file(list_dir, split)
            if list_file is not None:
                ext = self._infer_extension(pre_dir)
                ids = self._read_split_ids(list_file)
                self.pre_files = [os.path.join(pre_dir, self._ensure_ext(sid, ext)) for sid in ids]
                self.post_files = [os.path.join(post_dir, self._ensure_ext(sid, ext)) for sid in ids]
                self.label_files = [os.path.join(label_dir, self._ensure_ext(sid, ext)) for sid in ids]
            else:
                self.pre_files = self._list_images(pre_dir)
                self.post_files = self._list_images(post_dir)
                self.label_files = self._list_images(label_dir)
        else:
            raise FileNotFoundError(
                "Dataset directory not found. Expected either "
                f"{split_pre_dir} or {flat_pre_dir}."
            )

        if len(self.pre_files) == 0:
            raise ValueError(f"No images found in: {pre_dir}")

        if not (len(self.pre_files) == len(self.post_files) == len(self.label_files)):
            raise ValueError(
                "Dataset size mismatch: "
                f"pre={len(self.pre_files)} post={len(self.post_files)} label={len(self.label_files)}"
            )

    def __len__(self) -> int:
        return len(self.pre_files)

    def __getitem__(self, idx: int):
        pre = self._read_rgb(self.pre_files[idx])
        post = self._read_rgb(self.post_files[idx])
        mask = self._read_mask(self.label_files[idx])

        pre = self._resize(pre, self.img_size, is_mask=False)
        post = self._resize(post, self.img_size, is_mask=False)
        mask = self._resize(mask, self.img_size, is_mask=True)

        if self.transform is not None:
            pre, post, mask = self.transform(pre, post, mask)

        pre_t = torch.from_numpy(pre).permute(2, 0, 1).float()
        post_t = torch.from_numpy(post).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return {"pre": pre_t, "post": post_t, "mask": mask_t}

    def _list_images(self, folder: str) -> List[str]:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
        files.sort()
        return files

    def _infer_extension(self, folder: str) -> str:
        exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        for ext in exts:
            matches = [f for f in os.listdir(folder) if f.lower().endswith(ext)]
            if len(matches) > 0:
                return ext
        raise ValueError(f"No supported image extensions found in: {folder}")

    def _find_list_file(self, list_dir: str, split: str) -> Optional[str]:
        if not os.path.isdir(list_dir):
            return None
        candidates = [
            os.path.join(list_dir, f"{split}.txt"),
            os.path.join(list_dir, f"{split}.list"),
            os.path.join(list_dir, f"{split}.lst"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _read_split_ids(self, list_file: str) -> List[str]:
        with open(list_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f.readlines() if line.strip()]
        if len(ids) == 0:
            raise ValueError(f"No sample ids found in: {list_file}")
        return ids

    def _ensure_ext(self, name: str, ext: str) -> str:
        if name.lower().endswith(ext):
            return name
        return f"{name}{ext}"

    def _read_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _read_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        mask = (mask > 0).astype(np.float32)
        return mask

    def _resize(self, img: np.ndarray, size: Tuple[int, int], is_mask: bool) -> np.ndarray:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(img, (size[1], size[0]), interpolation=interp)
