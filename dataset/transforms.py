import random
from typing import Iterable, Tuple

import cv2
import numpy as np


class Compose:
    def __init__(self, transforms: Iterable):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        for t in self.transforms:
            pre, post, mask = t(pre, post, mask)
        return pre, post, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            pre = np.flip(pre, axis=1)
            post = np.flip(post, axis=1)
            mask = np.flip(mask, axis=1)
        return pre.copy(), post.copy(), mask.copy()


class RandomTemporalSwap:
    """Randomly swaps the pre and post images. This teaches the network that 
    Change Detection is symmetrical (A->B change is the same as B->A change)."""
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            return post.copy(), pre.copy(), mask
        return pre, post, mask



class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            pre = np.flip(pre, axis=0)
            post = np.flip(post, axis=0)
            mask = np.flip(mask, axis=0)
        return pre.copy(), post.copy(), mask.copy()


class RandomRotate90:
    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            pre = np.rot90(pre, k, axes=(0, 1))
            post = np.rot90(post, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        return pre.copy(), post.copy(), mask.copy()


class RandomBrightnessContrast:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, p: float = 0.5) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.p = float(p)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        if random.random() < self.p:
            alpha_pre = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta_pre = random.uniform(-self.brightness, self.brightness)
            pre = np.clip(pre * alpha_pre + beta_pre, 0.0, 1.0)
            
        if random.random() < self.p:
            alpha_post = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta_post = random.uniform(-self.brightness, self.brightness)
            post = np.clip(post * alpha_post + beta_post, 0.0, 1.0)
            
        return pre, post, mask


class GaussianBlur:
    def __init__(self, p: float = 0.2, ksize: int = 3, sigma: float = 0.3) -> None:
        self.p = float(p)
        self.ksize = int(ksize)
        self.sigma = float(sigma)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        if random.random() >= self.p:
            return pre, post, mask
        k = max(3, self.ksize)
        if k % 2 == 0:
            k += 1
        pre = cv2.GaussianBlur(pre, (k, k), self.sigma)
        post = cv2.GaussianBlur(post, (k, k), self.sigma)
        return pre, post, mask


class Normalize:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        pre = (pre - self.mean) / self.std
        post = (post - self.mean) / self.std
        return pre, post, mask


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0.05,
        contrast: float = 0.05,
        saturation: float = 0.05,
        hue: float = 0.02,
        p: float = 0.5,
    ) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.p = float(p)

    def __call__(self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray):
        # Independently apply color jitter to pre and post
        if random.random() < self.p:
            ops_pre = [
                lambda x: self._adjust_brightness(x),
                lambda x: self._adjust_contrast(x),
                lambda x: self._adjust_saturation_hue(x),
            ]
            random.shuffle(ops_pre)
            for op in ops_pre:
                pre = op(pre)
                
        if random.random() < self.p:
            ops_post = [
                lambda x: self._adjust_brightness(x),
                lambda x: self._adjust_contrast(x),
                lambda x: self._adjust_saturation_hue(x),
            ]
            random.shuffle(ops_post)
            for op in ops_post:
                post = op(post)
                
        return pre, post, mask

    def _adjust_brightness(self, img: np.ndarray) -> np.ndarray:
        delta = random.uniform(-self.brightness, self.brightness)
        return np.clip(img + delta, 0.0, 1.0)

    def _adjust_contrast(self, img: np.ndarray) -> np.ndarray:
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        return np.clip(img * alpha, 0.0, 1.0)

    def _adjust_saturation_hue(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_scale = 1.0 + random.uniform(-self.saturation, self.saturation)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
        hue_delta = random.uniform(-self.hue, self.hue) * 180.0
        hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180.0
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0
