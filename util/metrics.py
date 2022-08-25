import numpy as np
from skimage import measure


def tensor2im(image_tensor):
    image_numpy = image_tensor[0].numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    return np.around(image_numpy).astype(np.uint8)


def psnr(img1, img2):
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)
    # skimage.metrics.peak_signal_noise_ratio
    return measure.compare_psnr(img1, img2, 255)


def ssim(img1, img2):
    img1 = tensor2im(img1)
    img2 = tensor2im(img2)
    # skimage.metrics.structural_similarity
    return measure.compare_ssim(img1, img2, data_range=255, multichannel=True)


# 可视化实现