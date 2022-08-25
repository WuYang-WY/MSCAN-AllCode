import cv2
import os
from skimage import measure
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=r'/Users/prim/Downloads/GOPRO_results/GoPro')
parser.add_argument('--gt', type=str, default=r'/Users/prim/Downloads/GOPRO_Large/test')
parser.add_argument('--name', type=str, default='Gopro', choices=['Gopro', 'DeepVideo'])
args = parser.parse_args()


def psnr(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    [h1, w1, _] = img1.shape
    [h2, w2, _] = img2.shape
    if h1 != h2:
        img2 = cv2.resize(img2, (w1, h1))
    # return metrics.peak_signal_noise_ratio(img1, img2, 255)
    return measure.compare_psnr(img1, img2, 255)


def ssim(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    [h1, w1, _] = img1.shape
    [h2, w2, _] = img2.shape
    if h1 != h2:
        img2 = cv2.resize(img2, (w1, h1))
    # return metrics.structural_similarity(img1, img2, data_range=255, multichannel=True)
    return measure.compare_ssim(img1, img2, data_range=255, multichannel=True)


def calc(src, label, p):
    dirs = os.listdir(src)
    print(dirs)
    psnr_total = 0.0
    ssim_total = 0.0
    count = 0
    for item in dirs:
        print(item)
        if '.DS_Store' in os.path.join(src, item):
            print('FIND DSSTORE')
            continue
        blur_list = os.listdir(os.path.join(src, item))
        psnr_file = '{}/{}.txt'.format(src, item)
        with open(psnr_file, 'w') as f:
            _psnr = 0.0
            _ssim = 0.0
            _count = 0
            for img_name in blur_list:
                p1 = os.path.join(src, item, img_name)
                img_name=img_name.replace('_x1_SR','')#用完请删除此句 
                p2 = os.path.join(label, item, p, img_name)
                print('input:{}'.format(p1))
                psnr_v = psnr(p1, p2)
                ssim_v = ssim(p1, p2)
                print('img_name:{}\tpsnr:{}\tssim:{}'.format(img_name, psnr_v, ssim_v))
                f.write('img_name:{}\tpsnr:{}\tssim:{}\n'.format(img_name, psnr_v, ssim_v))
                _psnr += psnr_v
                _ssim += ssim_v
                _count += 1
            psnr_total += _psnr
            ssim_total += _ssim
            count += _count
            _psnr /= _count
            _ssim /= _count
            f.write('mean psnr:{}\tssim:{}\n'.format(_psnr, _ssim))
    psnr_total /= count
    ssim_total /= count
    f = open('{}/total_psnr.txt'.format(src), 'w')
    print('mean psnr:{}\tssim:{}'.format(psnr_total, ssim_total))
    f.write('mean psnr:{}\tssim:{}\n'.format(psnr_total, ssim_total))
    f.close()


if __name__ == '__main__':
    src = args.input
    label = args.gt
    p = ['sharp', 'GT']
    name = args.name
    tmp = p[0]
    if name == 'Gopro':
        tmp = p[0]
    elif name == 'DeepVideo':
        tmp = p[1]
    else:
        print('请指定正确的名字, 两个文件名分别为:[Gopro|DeepVideo]')
        exit(1)
    calc(src, label, tmp)
