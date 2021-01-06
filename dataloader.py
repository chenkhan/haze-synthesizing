import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
from numpy import random
from util.image_folder import make_dataset
import torch

class TestDataset(data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()

    def name(self):
        return 'TestDataset'

    def initialize(self,opt):
        self.opt=opt
        self.dir_gt=os.path.join(opt.dataroot,'a/')
        self.gt_paths = make_dataset(self.dir_gt)
        self.gt_paths = sorted(self.gt_paths)
        self.dir_depth=os.path.join(opt.dataroot,'b/')
        self.d_paths = make_dataset(self.dir_depth)
        self.d_paths = sorted(self.d_paths)
        transform_list1 = [transforms.ToTensor()]
        transform_list2 = [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list3 = [transforms.Normalize(0.5, 0.5)]
        self.transform1 = transforms.Compose(transform_list1)
        self.transform2 = transforms.Compose(transform_list2)
        self.transform3 = transforms.Compose(transform_list3)

    def __getitem__(self, index):
        self.A_path = self.gt_paths[index]
        self.B_path = self.d_paths[index]
        A = Image.open(self.A_path).convert('RGB')
        A_size = A.size
        B = Image.open(self.B_path).convert('L')
        A = A.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        B = B.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        A = self.transform1(A)
        B = self.transform1(B)
        A = self.transform2(A)
        B = self.transform3(B)
        return {
            'A': A,
            'A_size': A_size,
            'B': B,
            'A_name': self.A_path.split('/')[-1],
            'B_name': self.B_path.split('/')[-1],
            'A_path': self.A_path,
            'B_path': self.B_path}
    def __len__(self):
        return len(self.gt_paths)

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def name(self):
        return 'Dataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_sgt = os.path.join(opt.dataroot, 'a/')
        self.sgt_paths = make_dataset(self.dir_sgt)
        self.sgt_paths = sorted(self.sgt_paths)
        self.dir_sdepth = os.path.join(opt.dataroot, 'b/')
        self.sdepth_paths = make_dataset(self.dir_sdepth)
        self.sdepth_paths = sorted(self.sdepth_paths)
        self.transform = get_transform(opt)
        self.dir_rgt = os.path.join(opt.dataroot, 'c/')
        self.rgt_paths = make_dataset(self.dir_rgt)
        self.rgt_paths = sorted(self.rgt_paths)
        self.dir_uhz = os.path.join(opt.dataroot, 'd/')
        self.uhz_paths = make_dataset(self.dir_uhz)
        self.uhz_paths = sorted(self.uhz_paths)
        self.dir_rhz = os.path.join(opt.dataroot, 'e/')
        self.rhz_paths = make_dataset(self.dir_rhz)
        self.rhz_paths = sorted(self.rhz_paths)
        transform_list1 = [transforms.ToTensor()]
        transform_list2 = [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list3 = [transforms.Normalize(0.5, 0.5)]
        self.transform1 = transforms.Compose(transform_list1)
        self.transform2 = transforms.Compose(transform_list2)
        self.transform3 = transforms.Compose(transform_list3)

    def __getitem__(self, index):
        # A:S_GT B:S_DEPTH C:R_GT E:R_HZY
        c_ind=random.randint(0, len(self.rgt_paths))
        d_ind=random.randint(0, len(self.uhz_paths))
        self.A_path = self.sgt_paths[index]
        self.B_path = self.sdepth_paths[index]
        self.C_path = self.rgt_paths[c_ind]
        self.D_path = self.uhz_paths[d_ind]
        self.E_path = self.rhz_paths[c_ind]
        A = Image.open(self.A_path).convert('RGB')
        B = Image.open(self.B_path).convert('L')
        C = Image.open(self.C_path).convert('RGB')
        D = Image.open(self.D_path).convert('RGB')
        E = Image.open(self.E_path).convert('RGB')
        A = A.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        B = B.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        A = self.transform1(A)
        B = self.transform1(B)
        C = C.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        C = self.transform1(C)
        D = D.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        D = self.transform1(D)
        E = E.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        E = self.transform1(E)
        w = A.size(2)
        h = A.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]
        C = C[:, h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]
        D = D[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        E = E[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            C = C.index_select(2, idx)
            D = D.index_select(2, idx)
            E = E.index_select(2, idx)
        A = self.transform2(A)
        C = self.transform2(C)
        B = self.transform3(B)
        D = self.transform2(D)
        E = self.transform2(E)
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'E': E,
            'E_paths': self.E_path,
            'D_paths': self.D_path,
            'C_paths': self.C_path,
            'A_paths': self.A_path,
            'B_paths': self.B_path}

    def __len__(self):
        return len(self.sgt_paths)


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSizeX, opt.loadSizeY]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSizeX)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if opt.phase == 'test':
        transform_list = []
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        if(opt.isTrain):
            self.dataset = Dataset()
        else:
            self.dataset = TestDataset()
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader