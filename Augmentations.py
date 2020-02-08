import cv2
import random
import numpy as np
from albumentations import ShiftScaleRotate, Blur

crop_size = "Full"


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t.apply(x[0], mask=x[1])

        return x


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply(self, img, mask):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        return (img, mask)


class RScale:
    def __init__(self, prob_scale=0.5, prob_rotate=0.5):
        self.prob_scale = prob_scale
        self.prob_rotate = prob_rotate
        self.scale = ShiftScaleRotate(
            rotate_limit=45, shift_limit=0.1, scale_limit=0.1, p=1
        )

    def apply(self, img, mask):

        # scale
        if random.random() < self.prob_scale:

            angles = 0
            rand_scale = 0.75 + 0.75 * random.random()

            img = self.scale.apply(img, angle=angles, scale=rand_scale, dx=0, dy=0)
            mask = self.scale.apply_to_mask(
                mask, angle=angles, scale=rand_scale, dx=0, dy=0
            )

        # rotate
        if random.random() < self.prob_rotate:

            scales = 1
            rand_angle = random.randint(-45, 45)

            img = self.scale.apply(img, angle=rand_angle, scale=scales, dx=0, dy=0)
            mask = self.scale.apply_to_mask(
                mask, angle=rand_angle, scale=scales, dx=0, dy=0
            )

        return (img, mask)


class Transpose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply(self, img, mask):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            mask = mask.transpose(1, 0, 2)
        return (img, mask)


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class RandomGamma:
    def __init__(self, limit=0.2, prob=0.5):
        self.limit = limit
        self.prob = prob

    def apply(self, img, mask):
        if random.random() < self.prob:
            gamma = 1.0 + self.limit * random.uniform(-1, 1)

            img = img ** (1.0 / gamma)
            img = np.clip(img, 0, 1)

        return (img, mask)


class Resize:
    def __init__(self):
        pass

    def apply(self, img, mask):

        img = cv2.resize(img, (480, 320))
        mask = cv2.resize(mask, (480, 320))

        if crop_size == "Full":
            return (img, mask)

        return (img, mask)


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply(self, img, mask):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        return (img, mask)


class RBlur:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.blur = Blur(blur_limit=7, p=1)

    def apply(self, img, mask):

        if random.random() < self.prob:
            img = self.blur.apply(img)

        return (img, mask)


class RandomBrightnessContrast:
    def __init__(self, limit_alpha=0.4, limit_beta=0.4, prob=0.1):
        self.limit_alpha = limit_alpha
        self.limit_beta = limit_beta
        self.prob = prob

    def apply(self, img, mask, alpha=1.0, beta=0.0):
        if random.random() < self.prob:
            alpha = 1.0 + random.uniform(-self.limit_alpha, self.limit_alpha)
        if random.random() < self.prob:
            beta = 0.0 + random.uniform(-self.limit_beta, self.limit_beta)

            img = np.clip(alpha * img + beta * np.mean(img), 0, 1)
        return (img, mask)


train_transforms = DualCompose(
    [
        Resize(),
        HorizontalFlip(0.5),
        VerticalFlip(0.5),
        RandomBrightnessContrast(prob=0.35),
        RandomGamma(prob=0.35),
        RScale(prob_scale=0.3, prob_rotate=0.3),
        RBlur(prob=0.3),
    ]
)

test_transforms = DualCompose([Resize()])
