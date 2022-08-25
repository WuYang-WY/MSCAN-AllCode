from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as F
from PIL import Image
import os
import numpy as np
import random
from util.json_entity import JsonObj


class GoProDataset(Dataset):
    def __init__(self,
                 blur_image_files,
                 sharp_image_files,
                 root_dir,
                 crop=False,
                 crop_size=256,
                 multi_scale=False,
                 rotation=False,
                 color_augment=False,
                 transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)
        self.rotate45 = transforms.RandomRotation(45)

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        blur_image = Image.open(os.path.join(self.root_dir, image_name[0],
                                             image_name[1], image_name[2], image_name[3])).convert('RGB')
        sharp_image = Image.open(os.path.join(self.root_dir, image_name[0],
                                              image_name[1], 'sharp', image_name[3])).convert('RGB')
        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree)
            sharp_image = transforms.functional.rotate(sharp_image, degree)
        if self.color_augment:
            # contrast_factor = 1 + (0.2 - 0.4*np.random.rand())
            # blur_image = transforms.functional.adjust_contrast(blur_image, contrast_factor)
            # sharp_image = transforms.functional.adjust_contrast(sharp_image, contrast_factor)
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)
        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)
        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2]
            Ws = np.random.randint(0, W - self.crop_size - 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size - 1, 1)[0]
            blur_image = blur_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            sharp_image = sharp_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
        if self.multi_scale:
            H = sharp_image.size()[1]
            W = sharp_image.size()[2]
            blur_image_s1 = transforms.ToPILImage()(blur_image)
            sharp_image_s1 = transforms.ToPILImage()(sharp_image)
            blur_image_s2 = transforms.ToTensor()(transforms.Resize([H / 2, W / 2])(blur_image_s1))
            sharp_image_s2 = transforms.ToTensor()(transforms.Resize([H / 2, W / 2])(sharp_image_s1))
            blur_image_s3 = transforms.ToTensor()(transforms.Resize([H / 4, W / 4])(blur_image_s1))
            sharp_image_s3 = transforms.ToTensor()(transforms.Resize([H / 4, W / 4])(sharp_image_s1))
            blur_image_s1 = transforms.ToTensor()(blur_image_s1)
            sharp_image_s1 = transforms.ToTensor()(sharp_image_s1)
            return {'blur_image_s1': blur_image_s1, 'blur_image_s2': blur_image_s2, 'blur_image_s3': blur_image_s3,
                    'sharp_image_s1': sharp_image_s1, 'sharp_image_s2': sharp_image_s2,
                    'sharp_image_s3': sharp_image_s3}
        else:
            return {'blur_image': blur_image,
                    'sharp_image': sharp_image,
                    'dir': image_name[1],
                    'image_name': image_name[3]}


class GoProDatasetWithDA(Dataset):
    '''
    dataset with data augmentation including noise, rotate
    '''

    def __init__(self,
                 blur_image_files,
                 sharp_image_files,
                 root_dir,
                 crop=False,
                 crop_size=256,
                 noise=False,
                 rotation=False):
        print('=================> use dataset with data augmentation <================')
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.crop = crop
        self.crop_size = crop_size
        self.noise = noise
        self.rotation = rotation

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        blur_image = Image.open(os.path.join(self.root_dir, image_name[0],
                                             image_name[1], image_name[2], image_name[3])).convert('RGB')
        sharp_image = Image.open(os.path.join(self.root_dir, image_name[0],
                                              image_name[1], 'sharp', image_name[3])).convert('RGB')
        if self.noise:  # 添加1%的高斯噪声
            rate = np.random.random()
            if rate > 0.7:  # 当概率大于0.7时添加噪声
                noise_img = np.array(blur_image)
                noise_img = noise_img + np.random.normal(0, 0.01 * 255, noise_img.shape)
                noise_img = np.clip(noise_img, 0, 255).astype('uint8')
                blur_image = Image.fromarray(noise_img).convert('RGB')
        if self.rotation:  # 旋转
            rate = np.random.random()
            if rate > 0.7:
                # 0, 1, 2分别代表90, 180, 270度
                degree = np.random.choice([90, 180, 270])
                blur_image = blur_image.rotate(degree)
                sharp_image = sharp_image.rotate(degree)
        blur_tensor = F.to_tensor(blur_image)  # 转换为tensor
        sharp_tensor = F.to_tensor(sharp_image)
        if self.crop:
            h = blur_tensor.size(1)
            w = blur_tensor.size(2)
            Ws = np.random.randint(0, w - self.crop_size - 1, 1)[0]#randint取1个随机数
            Hs = np.random.randint(0, h - self.crop_size - 1, 1)[0]
            blur_tensor = blur_tensor[:, Hs:Hs + self.crop_size, Ws:Ws + self.crop_size]
            sharp_tensor = sharp_tensor[:, Hs:Hs + self.crop_size, Ws:Ws + self.crop_size]
        return {'blur_image': blur_tensor,
                'sharp_image': sharp_tensor,
                'dir': image_name[1],
                'image_name': image_name[3]}


class DeepVideoDataset(Dataset):

    def __init__(self,
                 root_path,
                 txt_path,
                 phase='train',
                 crop=False,
                 crop_size=256,
                 transform=None):
        self._path = txt_path
        self._root_path = root_path
        obj = JsonObj()
        obj.load_json(txt_path)
        self.crop = crop
        self.crop_size = crop_size
        self.transform = transform
        self.blur_list, self.gt_list = self._get_file_list(obj, phase)

    def _get_file_list(self, obj, phase):
        blur_list = []
        gt_list = []
        for item in obj._items:
            blur_path = os.path.join(self._root_path, item['name'], 'input')
            gt_path = os.path.join(self._root_path, item['name'], 'GT')
            blur_samples = [os.path.join(blur_path, img_name) for img_name in item['sample']]
            gt_samples = [os.path.join(gt_path, img_name) for img_name in item['sample']]
            if item['phase'] == phase:
                blur_list.extend(blur_samples)
                gt_list.extend(gt_samples)
        return blur_list, gt_list

    def __len__(self):
        return len(self.blur_list)

    def __getitem__(self, item):
        blur_path = self.blur_list[item]
        gt_path = self.gt_list[item]
        blur_img = Image.open(blur_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        if self.transform:
            blur_img = self.transform(blur_img)
            gt_img = self.transform(gt_img)
        if self.crop:
            W = blur_img.size()[1]
            H = gt_img.size()[2]
            Ws = np.random.randint(0, W - self.crop_size - 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size - 1, 1)[0]
            blur_img = blur_img[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            gt_img = gt_img[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
        return {'blur_image': blur_img, 'sharp_image': gt_img}
