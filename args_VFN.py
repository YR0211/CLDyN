
import argparse

parser = argparse.ArgumentParser(description='**CLDyN-VFN**')
parser.add_argument('--train_save_img_dir', default='./checkpoints/images', type=str)
parser.add_argument('--train_save_model_dir', default='./checkpoints/train_models', type=str)
parser.add_argument('--test_checkpoint_path', default='...', type=str)
parser.add_argument('--test_dataset_path', default='...', type=str)
parser.add_argument('--save_image_num', default=8, type=int)
parser.add_argument('--save_model_num', default=20, type=int)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
parser.add_argument('--LR', type=float, default=0.001)
parser.add_argument('--Epoch', type=float, default=100)
args = parser.parse_args()