from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from util.datasets import GoProDataset, DeepVideoDataset, GoProDatasetWithDA
from torch.utils.data import DataLoader
from torchvision import transforms
import util.metrics as metrics
import torchvision
import os
import torch
from util.util import PlotPSNR, SaveBestPSNR
import util.util as util
import math
import time
import torch.nn.functional as F
from PIL import Image
from util.json_entity import JsonObj


class Base(object):

    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.model = None
        self.loss_fn = None
        self.lr = None
        self.optimizer = None
        self.lr_decay = None
        # save result
        self.output = None
        # save max value calculated
        self.loss_val = None
        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size
        self.pre_trained = self.args.pre_trained
        self.root_dir = self.args.root_dir
        self.crop_size = self.args.crop_size
        self.exp_dir = self.args.exp_dir  # the dir of save model and test result
        self.save_model_name = self.args.save_model_name
        self.best_psnr = 0.
        self.best_epoch = 0
        self.step = self.args.step  # the interval between tests
        self.plot_util = PlotPSNR(step=self.step, exp_dir=self.exp_dir)
        self.save_psnr = SaveBestPSNR(exp_dir=self.exp_dir)

    def trainer_initial(self, model, loss_fn, step_size=1000):
        self.model = model
        self.model.apply(Base.weight_init).cuda()
        self.lr = self.args.learning_rate
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)
        self.lr_decay = StepLR(self.optimizer, step_size)
        self.loss_fn = loss_fn

    @staticmethod
    def weight_init(m):
        classname = m.__class__.__name__  # DCCANShareTwoLevelPlusWithSkipNoSPP
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

    def get_model_out(self, blur):
        '''
        returns the result of the model, and when the result of the model has three scales,
        returns the result of the largest scale.
        the function only be invoked during testing
        :param blur: blur image
        :return:
        '''
        return self.model(blur)

    def forward(self, blur, gt):
        self.output = self.model(blur)
        self.loss_val = self.loss_fn(self.output, gt)

    def backward(self):
        self.optimizer.zero_grad()
        self.loss_val.backward()
        # gradients clip
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)  # 梯度剪裁
        self.optimizer.step()

    def eval_step(self, input_path, output_path, image_name):
        path = os.path.join(input_path, image_name)
        B1 = transforms.ToTensor()(Image.open(path).convert('RGB'))
        B1 = (B1 - 0.5).unsqueeze(0).cuda()
        # make sure the picture is a multiple of 16
        [b, c, h, w] = B1.shape
        new_h = h - h % 16
        new_w = w - w % 16
        B1 = F.interpolate(B1, size=(new_h, new_w), mode='bilinear')
        start = time.time()
        deblur = self.get_model_out(B1).cpu() + 0.5
        duration = time.time() - start
        print('image:{}\ttime:{:.4f}'.format(image_name, duration))
        path = os.path.join(output_path, image_name)
        torchvision.utils.save_image(deblur.data, path)

    def eval(self):
        epoch = self.restore_model(best=self.args.use_best)
        print(' best model in epoch:{}'.format(epoch))
        input_path = self.args.gopro_input_path
        dirs = os.listdir(input_path)
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        for item in dirs:
            dir = os.path.join(input_path, item, 'blur')
            images = os.listdir(dir)
            output_path = os.path.join(self.args.output_path, item)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for image_name in images:
                with torch.no_grad():
                    self.eval_step(dir, output_path, image_name)

    def eval_deepvideo_real(self):
        epoch = self.restore_model(best=self.args.use_best)
        print(' best model in epoch:{}'.format(epoch))
        input_path = self.args.deepvideo_input_path
        dirs = os.listdir(input_path)
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        for item in dirs:
            dir = os.path.join(input_path, item, 'input')
            images = os.listdir(dir)
            output_path = os.path.join(self.args.output_path, item)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for image_name in images:
                with torch.no_grad():
                    self.eval_step(dir, output_path, image_name)

    def eval_deep_video(self):
        epoch = self.restore_model(best=self.args.use_best)
        print(' best model in epoch:{}'.format(epoch))
        obj = JsonObj()
        obj.load_json('datas/DeepVideo/DeepVideo.json')
        input_path = self.args.deepvideo_input_path
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        for item in obj._items:
            if item['phase'] == 'test':
                dir = os.path.join(input_path, item['name'], 'input')
                images = item['sample']
                output_path = os.path.join(self.args.output_path, item['name'])
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                for image_name in images:
                    with torch.no_grad():
                        self.eval_step(dir, output_path, image_name)

    def eval_real_img(self):
        util.print_model(self.model)  # calc model size
        epoch = self.restore_model(best=self.args.use_best)
        print(' best model in epoch:{}'.format(epoch))
        input_path = self.args.real_input_path
        images = os.listdir(input_path)
        output_path = self.args.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image_name in images:
            with torch.no_grad():
                self.eval_step(input_path, output_path, image_name)

    def load_pre_trained_weight(self):
        model_dict = self.model.state_dict()
        pre_path = os.path.join(self.args.exp_dir, 'pre_trained.pth')
        assert os.path.exists(pre_path), 'please specify a pre-trained model'
        pre_model_dict = torch.load(pre_path)
        epoch = pre_model_dict['epoch']
        print('the last epoch in pre-trained model:{}'.format(epoch))
        pre_model = pre_model_dict['model']
        pre_model = {k: v for k, v in pre_model.items() if k in model_dict}
        model_dict.update(pre_model)
        self.model.load_state_dict(model_dict)
        return 0

    def restore_model(self, best=False):
        if self.pre_trained:
            return self.load_pre_trained_weight()
        path = os.path.join(self.exp_dir, '{}.pth'.format(self.save_model_name))
        if best:
            path = os.path.join(self.exp_dir, 'best_{}.pth'.format(self.save_model_name))
        if not os.path.exists(path):
            return 0
        model_dict = torch.load(path)
        self.optimizer.load_state_dict(model_dict['optimizer'])
        epoch = model_dict['epoch']
        self.model.load_state_dict(model_dict['model'])
        self.lr_decay.load_state_dict(model_dict['lr_decay'])
        self.best_psnr, self.best_epoch = self.save_psnr.load()
        self.plot_util.load()
        return epoch

    def save_model(self, epoch, best=False):
        model_dict = {'epoch': epoch,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_decay': self.lr_decay.state_dict()}
        path = os.path.join(self.exp_dir, '{}.pth'.format(self.save_model_name))
        if best:
            path = os.path.join(self.exp_dir, 'best_{}.pth'.format(self.save_model_name))
            self.save_psnr.save(self.best_psnr, self.best_epoch)
        torch.save(model_dict, path)
        self.plot_util.save()
        self.plot_util.write_to_csv()

    def save_net_output(self, images, iter, epoch):
        dir = os.path.join(self.exp_dir, 'test_result', '{}'.format(epoch))
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_path = os.path.join(dir, '{}.png'.format(iter))
        torchvision.utils.save_image(images, file_path)

    def train_step(self, data_loader, epoch):
        for iter, images in enumerate(data_loader):
            blur = (images['blur_image'] - 0.5).cuda()
            gt = (images['sharp_image'] - 0.5).cuda()
            self.forward(blur, gt)
            self.backward()
            # output loss value to console
            if not iter % 10:
                print('epoch:{}\titer:{}'
                      '\tloss:{}\tlr:{}'
                      .format(epoch, iter,
                              self.loss_val.item(),
                              self.lr_decay.get_lr()[0]))

    def eval_in_training(self, data_loader, epoch):
        psnr_sum = 0.
        ssim_sum = 0.
        for iter, image in enumerate(data_loader):
            with torch.no_grad():
                blur = (image['blur_image'] - 0.5).cuda()
                gt = image['sharp_image']
                deblur = self.get_model_out(blur).cpu() + 0.5
                p = metrics.psnr(deblur, gt)
                s = metrics.ssim(deblur, gt)
                print('psnr:{:.2f}\tssim:{:.4f}'.format(p, s))
                psnr_sum += p
                ssim_sum += s
                self.save_net_output(deblur, iter=iter, epoch=epoch)
        psnr_sum /= 20
        ssim_sum /= 20
        self.plot_util.plot(psnr_sum, epoch)
        # save best model
        if self.best_psnr < psnr_sum:
            self.best_psnr = psnr_sum
            self.best_epoch = epoch
            self.save_model(epoch, best=True)
        print('current epoch:{}\tmodel mean psnr:{:.2f}'
              '\tssim:{:.4f}\tbest psnr:{:.2f}\tepoch:{}'
              .format(epoch, psnr_sum, ssim_sum,
                      self.best_psnr, self.best_epoch))

    def train(self):
        assert self.model is not None, 'please define model!!'
        util.print_model(self.model)  # calc model size
        # train_dataset = GoProDataset(blur_image_files='datas/GoPro/train_blur_file.txt',
        #                              sharp_image_files='datas/GoPro/train_sharp_file.txt',
        #                              root_dir=self.root_dir,
        #                              crop=True,
        #                              crop_size=self.crop_size,
        #                              transform=transforms.Compose([
        #                                 transforms.ToTensor()
        #                                 ]))
        train_dataset = GoProDatasetWithDA(blur_image_files='datas/GoPro/train_blur_file.txt',
                                           sharp_image_files='datas/GoPro/train_sharp_file.txt',
                                           root_dir=self.root_dir,
                                           crop=True,
                                           crop_size=256,
                                           rotation=True,
                                           noise=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=16)
        test_dataset = GoProDataset(blur_image_files='datas/GoPro/test_blur_20.txt',
                                    sharp_image_files='datas/GoPro/test_sharp_20.txt',
                                    root_dir=self.root_dir,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
        start_epoch = self.restore_model()
        # start train
        for epoch in range(start_epoch, self.epoch):
            self.train_step(train_dataloader, epoch)
            self.lr_decay.step(epoch)  # update learning rate
            self.save_model(epoch)
            if not epoch % self.step:
                self.eval_in_training(test_dataloader, epoch)

    def train_deep_video(self):
        assert self.model is not None, 'please define model!!'
        print('start train deep video')
        util.print_model(self.model)  # calc model size
        train_dataset = DeepVideoDataset(root_path=self.root_dir,
                                         txt_path='datas/DeepVideo/DeepVideo.json',
                                         phase='train',
                                         crop=True,
                                         crop_size=self.crop_size,
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=16)
        test_dataset = DeepVideoDataset(root_path=self.root_dir,
                                        txt_path='datas/DeepVideo/deep_video_test.json',
                                        phase='test',
                                        transform=transforms.Compose([
                                            transforms.ToTensor()
                                        ]))
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
        start_epoch = self.restore_model()
        # start train
        for epoch in range(start_epoch, self.epoch):
            self.train_step(train_dataloader, epoch)
            self.lr_decay.step(epoch)  # update learning rate
            self.save_model(epoch)
            if not epoch % self.step:
                self.eval_in_training(test_dataloader, epoch)
