import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from PIL import Image
from kornia.geometry import vflip, hflip

from yolo.utils.general import segments2boxes


class TrainDataset(data.Dataset):
    def __init__(self, vis_dir_list, ir_dir_list, transform, crop_size=None, img_type='L'):
        super(TrainDataset, self).__init__()
        if crop_size is None:
            crop_size = [256, 256]
        assert img_type == 'RGB' or img_type == 'L'
        assert len(vis_dir_list) == len(ir_dir_list)
        self.img_type = img_type
        self.crop = torchvision.transforms.RandomCrop(crop_size)
        self.dataset_nums = len(vis_dir_list)

        self.vis_paths_list = []
        self.ir_paths_list = []
        self.dataset_indexs = []

        for i, (vis_dir, ir_dir) in enumerate(zip(vis_dir_list, ir_dir_list)):
            _, vis_paths = self.find_file(vis_dir)
            _, ir_paths = self.find_file(ir_dir)
            assert len(vis_paths) == len(ir_paths)
            self.vis_paths_list.extend(vis_paths)
            self.ir_paths_list.extend(ir_paths)
            self.dataset_indexs.extend([i] * len(ir_paths))
            assert len(self.vis_paths_list) == len(self.ir_paths_list) == len(self.dataset_indexs)

        self.transform = transform

    def find_file(self, dir):
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                for file_name in os.listdir(os.path.join(dir, dir_name)):
                    paths.append(os.path.join(dir, file_name, file_name))
        else:
            paths = list(Path(dir).glob('*'))
        return path, paths

    def read_image(self, path):
        img = Image.open(str(path)).convert(self.img_type)
        img = self.transform(img)
        img = img.unsqueeze(dim=0)
        return img

    def __getitem__(self, index):

        vis_path = self.vis_paths_list[index]
        ir_path = self.ir_paths_list[index]

        vis_img = self.read_image(vis_path)
        ir_img = self.read_image(ir_path)

        vis_ir = torch.cat([vis_img, ir_img], dim=1)

        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)

        couple = self.crop(vis_ir)

        _, _, h, w = couple.size()
        couple_d2 = F.interpolate(couple, size=(h // 2, w // 2), mode='bilinear', align_corners=True)

        couple = couple.squeeze(dim=0)
        couple_d2 = couple_d2.squeeze(dim=0)

        if self.img_type == 'RGB':
            vis, ir = torch.split(couple, [3, 3], dim=0)
            vis_d2, ir_d2 = torch.split(couple_d2, [3, 3], dim=0)
            return vis, ir, vis_d2, ir_d2, str(vis_path)
        elif self.img_type == 'L':
            vis, ir = torch.split(couple, [1, 1], dim=0)
            vis_d2, ir_d2 = torch.split(couple_d2, [1, 1], dim=0)
            return vis, ir, vis_d2, ir_d2, str(vis_path)

    def __len__(self):
        return len(self.vis_paths_list)


class TrainDataset_od(data.Dataset):
    def __init__(self, vis_dir_list, ir_dir_list, label_dir_list, flip_ud=0.0, flip_lr=0.5, img_size=None,
                 img_type='RGB'):
        super(TrainDataset_od, self).__init__()
        if img_size is None:
            img_size = [640, 640]
        assert img_type == 'RGB' or img_type == 'L'
        assert len(vis_dir_list) == len(ir_dir_list)
        self.img_type = img_type
        self.toTensor = torchvision.transforms.ToTensor()
        self.Resize = torchvision.transforms.Resize(img_size)
        self.dataset_nums = len(vis_dir_list)
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr

        vis_paths_list = []
        ir_paths_list = []
        label_paths_list = []

        for i, (vis_dir, ir_dir, label_dir) in enumerate(zip(vis_dir_list, ir_dir_list, label_dir_list)):
            _, vis_paths = self.find_file(vis_dir)
            _, ir_paths = self.find_file(ir_dir)
            _, label_paths = self.find_file(label_dir)
            assert len(vis_paths) == len(ir_paths) == len(label_paths)
            vis_paths_list.extend(vis_paths)
            ir_paths_list.extend(ir_paths)
            label_paths_list.extend(label_paths)
            assert len(vis_paths_list) == len(ir_paths_list) == len(label_paths_list)

        self.vis_paths_list = sorted(vis_paths_list, key=lambda path: os.path.splitext(os.path.basename(path))[0])
        self.ir_paths_list = sorted(ir_paths_list, key=lambda path: os.path.splitext(os.path.basename(path))[0])
        self.label_paths_list = sorted(label_paths_list, key=lambda path: os.path.splitext(os.path.basename(path))[0])

        assert all(f1 == f2 == f3 for f1, f2, f3 in zip(
            [os.path.splitext(os.path.basename(path))[0] for path in self.vis_paths_list],
            [os.path.splitext(os.path.basename(path))[0] for path in self.ir_paths_list],
            [os.path.splitext(os.path.basename(path))[0] for path in self.label_paths_list]
        ))

    def find_file(self, dir):
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                for file_name in os.listdir(os.path.join(dir, dir_name)):
                    paths.append(os.path.join(dir, file_name, file_name))
        else:
            paths = list(Path(dir).glob('*'))
        return path, paths

    def read_image(self, path):
        img = Image.open(str(path)).convert(self.img_type)
        img = self.toTensor(img)
        img = img.unsqueeze(dim=0)
        return img

    def read_label(self, path):
        with open(path) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if any(len(x) > 6 for x in lb):
                classes = np.array([x[0] for x in lb], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
            lb = np.array(lb, dtype=np.float32)
        return lb

    def __getitem__(self, index):

        vis_path = self.vis_paths_list[index]
        ir_path = self.ir_paths_list[index]
        label_path = self.label_paths_list[index]

        vis_img = self.read_image(vis_path)
        ir_img = self.read_image(ir_path)
        label = self.read_label(label_path)

        t = torch.cat([vis_img, ir_img], dim=1)

        t = self.Resize(t)

        nl = len(label)

        if random.random() < self.flip_ud:
            t = vflip(t)
            if nl:
                label[:, 2] = 1 - label[:, 2]

        if random.random() < self.flip_lr:
            t = hflip(t)
            if nl:
                label[:, 1] = 1 - label[:, 1]

        label_out = torch.zeros((nl, 6))
        if nl:
            label_out[:, 1:] = torch.from_numpy(label)

        t = t.squeeze(dim=0)

        if self.img_type == 'RGB':
            vis, ir = torch.split(t, 3, dim=0)
        elif self.img_type == 'L':
            vis, ir = torch.split(t, 1, dim=0)

        return vis, ir, label_out, str(vis_path)

    def __len__(self):
        return len(self.vis_paths_list)

    @staticmethod
    def collate_fn(batch):
        vi, ir, label, path = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(vi, 0), torch.stack(ir, 0), torch.cat(label, 0), path


class RandomCrop(object):
    def __call__(self, ir, vi, mask, edge):
        H, W, _ = ir.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return ir[p0:p1, p2:p3, :], vi[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, ir, vi, mask, edge):
        if np.random.randint(2) == 0:
            return ir[:, ::-1, :], vi[:, ::-1, :], mask[:, ::-1], edge[:, ::-1]
        else:
            return ir, vi, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, ir, vi, mask):
        ir = cv2.resize(ir, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        vi = cv2.resize(vi, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return ir, vi, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, ir, vi, mask, edge=None):
        ir = (ir - self.mean) / self.std
        vi = (vi - self.mean) / self.std
        mask /= 255
        if edge is None:
            return ir, vi, mask
        else:
            edge /= 255
            return ir, vi, mask, edge


class ToTensor(object):
    def __call__(self, ir, vi, mask):
        ir = torch.from_numpy(ir)
        ir = ir.permute(2, 0, 1)
        vi = torch.from_numpy(vi)
        vi = vi.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return ir, vi, mask


class TrainDataset_sod(data.Dataset):
    def __init__(self, ir_dir, vi_dir, label_dir, edge_dir, mode='train'):
        self.mode = mode
        self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[56.77, 55.97, 57.50]]]))
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        self.toTensor_train = torchvision.transforms.ToTensor()

        if self.mode == 'train':
            _, ir_paths = self.find_file(ir_dir)
            _, vi_paths = self.find_file(vi_dir)
            _, label_paths = self.find_file(label_dir)
            _, edge_paths = self.find_file(edge_dir)
            self.ir_paths = sorted(ir_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            self.vi_paths = sorted(vi_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            self.label_paths = sorted(label_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            self.edge_paths = sorted(edge_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            assert all(f1 == f2 == f3 == f4 for f1, f2, f3, f4 in zip(
                [os.path.splitext(os.path.basename(path))[0] for path in self.ir_paths],
                [os.path.splitext(os.path.basename(path))[0] for path in self.vi_paths],
                [os.path.splitext(os.path.basename(path))[0] for path in self.label_paths],
                [os.path.splitext(os.path.basename(path))[0] for path in self.edge_paths]
            ))
        else:
            _, ir_paths = self.find_file(ir_dir)
            _, vi_paths = self.find_file(vi_dir)
            _, label_paths = self.find_file(label_dir)
            self.ir_paths = sorted(ir_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            self.vi_paths = sorted(vi_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            self.label_paths = sorted(label_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])
            assert all(f1 == f2 == f3 for f1, f2, f3 in zip(
                [os.path.splitext(os.path.basename(path))[0] for path in self.ir_paths],
                [os.path.splitext(os.path.basename(path))[0] for path in self.vi_paths],
                [os.path.splitext(os.path.basename(path))[0] for path in self.label_paths]
            ))

    def find_file(self, dir):
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                for file_name in os.listdir(os.path.join(dir, dir_name)):
                    paths.append(os.path.join(dir, file_name, file_name))
        else:
            paths = list(Path(dir).glob('*'))
        return path, paths

    def __getitem__(self, idx):
        ir_name = str(self.ir_paths[idx])
        vi_name = str(self.vi_paths[idx])
        label_name = str(self.label_paths[idx])
        ir = cv2.imread(ir_name)[:, :, ::-1].astype(np.float32)
        vi = cv2.imread(vi_name)[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(label_name, 0).astype(np.float32)
        shape = mask.shape

        if self.mode == 'train':
            edge_name = str(self.edge_paths[idx])
            edge = cv2.imread(edge_name, 0).astype(np.float32)
            _, _, mask, edge = self.normalize(ir, vi, mask, edge)
            ir, vi, mask, edge = self.randomcrop(ir, vi, mask, edge)
            ir, vi, mask, edge = self.randomflip(ir, vi, mask, edge)
            return ir, vi, mask, edge
        else:
            ir, vi, mask = self.normalize(ir, vi, mask)
            ir, vi, mask = self.resize(ir, vi, mask)
            ir, vi, mask = self.totensor(ir, vi, mask)
            return ir, vi, mask, shape, str(ir_name)

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        ir, vi, mask, edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            ir[i] = cv2.resize(ir[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            vi[i] = cv2.resize(vi[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i] = cv2.resize(edge[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        if self.mode == 'train':
            ir = torch.from_numpy(np.stack(ir, axis=0)).permute(0, 3, 1, 2) / 255.
            vi = torch.from_numpy(np.stack(vi, axis=0)).permute(0, 3, 1, 2) / 255.
        else:
            ir = torch.from_numpy(np.stack(ir, axis=0)).permute(0, 3, 1, 2)
            vi = torch.from_numpy(np.stack(vi, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return ir, vi, mask, edge

    def __len__(self):
        return len(self.ir_paths)


class TestDataset(data.Dataset):
    def __init__(self, vis_dir, ir_dir, transform, is_d2, img_type='L'):
        super(TestDataset, self).__init__()
        assert img_type == 'RGB' or img_type == 'L'
        self.is_d2 = is_d2
        self.img_type = img_type
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir

        self.vis_path, self.vis_paths = self.find_file(self.vis_dir)
        self.ir_path, self.ir_paths = self.find_file(self.ir_dir)

        assert (len(self.vis_path) == len(self.ir_path))

        self.transform = transform

    def find_file(self, dir):
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                for file_name in os.listdir(os.path.join(dir, dir_name)):
                    paths.append(os.path.join(dir, file_name, file_name))
        else:
            paths = list(Path(dir).glob('*'))
        return path, paths

    def read_image(self, path):
        img = Image.open(str(path)).convert(self.img_type)
        s = img.size
        img = self.transform(img)
        img = img.unsqueeze(dim=0)
        return img, s

    def __getitem__(self, index):
        vis_path = self.vis_paths[index]
        ir_path = self.ir_paths[index]

        vis, s = self.read_image(vis_path)
        ir, _ = self.read_image(ir_path)

        if self.is_d2:
            _, _, h, w = vis.size()
            vis_d2 = F.interpolate(vis, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
            ir_d2 = F.interpolate(ir, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
            vis_d2 = vis_d2.squeeze(dim=0)
            ir_d2 = ir_d2.squeeze(dim=0)

        vis = vis.squeeze(dim=0)
        ir = ir.squeeze(dim=0)

        if self.is_d2:
            return vis, ir, vis_d2, ir_d2, s, str(vis_path)
        else:
            return vis, ir, str(vis_path)

    def __len__(self):
        return len(self.vis_path)


def randrot(img):
    mode = np.random.randint(0, 4)
    return rot(img, mode)


def randfilp(img):
    mode = np.random.randint(0, 3)
    return flip(img, mode)


def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img


def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img