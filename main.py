import argparse
import sys
from trainer.trainer_factory import TrainerFactory


sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2000, help='the max iteration number')
parser.add_argument('--exp_dir', type=str, default=r'./experiment', help='the directory of experiment')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--root_dir', type=str, default=r'D:\dataset\GOPRO_Large', help='the directory of save data used training')
parser.add_argument('--crop_size', type=int, default=256, help='the image size used training')
parser.add_argument('--step', type=int, default=50, help='the interval between test')
parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--use_best', action='store_true', help='whether to use the best model in the test phase')
parser.add_argument('--pre_trained', action="store_true", help='whether to use the pre_trained model in the train phase')
parser.add_argument('--real_input_path', type=str, default=r'D:\dataset\real_image', help='real image path')
parser.add_argument('--gopro_input_path', type=str, default=r'D:\dataset\GOPRO_Large\test', help='gopro test image path')
parser.add_argument('--deepvideo_input_path', type=str, default=r'E:\data\DeepVideoDeblurring_Dataset\DeepVideoDeblurring_Dataset\qualitative_datasets', help='deep video test image path')
parser.add_argument('--output_path', type=str, default=r'./experimental/test_result', help='test result output path')
parser.add_argument('--save_model_name', type=str, default='dccan_model', help='the name of the saved model')
parser.add_argument('--trainer_name', type=str, default='TwoLevelPlusWithSkipConcat', help='optional model')
parser.add_argument('--eval_dataset_name',
                    type=str,
                    default='Gopro',
                    choices=['Gopro', 'DeepVideo', 'RealImg', 'DeepVideoReal'],
                    help='optional test dataset')

args = parser.parse_args()

if __name__ == '__main__':
    trainer_factory = TrainerFactory(args)
    trainer = trainer_factory.get_trainer(args.trainer_name)
    assert trainer is not None, 'please specify the correct name of the trainer'
    if args.phase == 'train':
        print('==============>start training....')
        trainer.train()
        # trainer.train_deep_video()
    else:
        print('==============>start testing {}....'.format(args.eval_dataset_name))
        if args.eval_dataset_name == 'Gopro':
            trainer.eval()
        elif args.eval_dataset_name == 'DeepVideo':
            trainer.eval_deep_video()
        elif args.eval_dataset_name == 'RealImg':
            trainer.eval_real_img()
        elif args.eval_dataset_name == 'DeepVideoReal':
            trainer.eval_deepvideo_real()
        else:
            print('please specify the correct name of the dataset')
