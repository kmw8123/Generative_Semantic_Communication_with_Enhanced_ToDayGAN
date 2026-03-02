import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import torch.nn.functional as F

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

        # Segmentation map 경로 설정 (예: segA/, segB/ 하위 폴더에 클래스 label map 저장)
        if self.opt.isTrain:
            self.seg_paths_A = sorted(make_dataset(os.path.join(opt.dataroot, 'segA'), self.opt.max_dataset_size))
            self.seg_paths_B = sorted(make_dataset(os.path.join(opt.dataroot, 'segB'), self.opt.max_dataset_size))
            self.seg_nc = opt.seg_nc  # segmentation 클래스 수 (예: 19)

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def load_segmentation(self, seg_path):
        seg = Image.open(seg_path).convert('L')  # [H, W] class index map
        seg = transforms.Resize((self.opt.fineSize, self.opt.fineSize), interpolation=Image.NEAREST)(seg)
        seg_tensor = transforms.ToTensor()(seg) * 255.0  # [1, H, W]
        seg_tensor = seg_tensor.squeeze().long()         # [H, W]
        one_hot = F.one_hot(seg_tensor, num_classes=self.seg_nc)  # [H, W, C]
        return one_hot.permute(2, 0, 1).float()           # [C, H, W]

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _ = self.load_image(DB, index_B)

            # Load segmentation maps
            seg_A = self.load_segmentation(self.seg_paths_A[index_A % len(self.seg_paths_A)])
            seg_B = self.load_segmentation(self.seg_paths_B[index_B % len(self.seg_paths_B)])

            bundle.update({
                'B': B_img,
                'DB': DB,
                'seg_A': seg_A,
                'seg_B': seg_B
            })

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'
