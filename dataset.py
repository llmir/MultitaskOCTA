from scipy.integrate._ivp.radau import P
import torch
import os
import random
import numpy as np
import cv2
import kornia
import kornia.augmentation as K

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
from scipy.ndimage import distance_transform_edt


def mean_and_std(paths):
    print('Calculating mean and std of training set for data normalization.')
    m_list, s_list = [], []
    for img_filename in tqdm(paths):
        img = cv2.imread(img_filename)
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    m = m[0][::-1][0]/255
    s = s[0][::-1][0]/255
    print(m)
    print(s)

    return m, s


class DatasetImageMaskContourDist(Dataset):

    # dataset_type(cup,disc,polyp),
    # distance_type(dist_mask,dist_contour,dist_signed)

    def __init__(self, file_names, distance_type, mean, std, clahe):

        self.file_names = file_names
        self.distance_type = distance_type
        self.mean = mean
        self.std = std
        self.clahe = clahe

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(img_file_name, self.mean, self.std, self.clahe)
        mask = load_mask(img_file_name)
        contour = load_contourheat(img_file_name)
        dist = load_distance(img_file_name, self.distance_type)
        cls = load_class(img_file_name)

        return img_file_name, image, mask, contour, dist, cls
        # return image, mask


def clahe_equalized(imgs):
    # print(imgs.shape)
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


def load_image(path, mean, std, clahe):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print(type(img))
    # if clahe:
    #     img = clahe_equalized(img)
    #     img = np.array(img ,dtype=np.float32)

    data_transforms = transforms.Compose(
        [
            # transforms.Resize(416),
            transforms.ToTensor(),
            # transforms.Normalize([0.445,], [0.222,]),
            transforms.Normalize([mean, ], [std, ]),
        ]
    )
    img = data_transforms(img)

    return img


def load_mask(path):

    mask = cv2.imread(path.replace("image", "mask").replace("png", "png"), 0)
    mask[mask == 255] = 1

    return torch.from_numpy(np.expand_dims(mask, 0)).long()


def load_contour(path):

    contour = cv2.imread(path.replace("image", "contour").replace("png", "png"), 0)
    contour[contour == 255] = 1

    return torch.from_numpy(np.expand_dims(contour, 0)).float()


def load_contourheat(path):

    path = path.replace("image", "contour").replace("png", "mat")
    contour = io.loadmat(path)["contour"]

    return torch.from_numpy(np.expand_dims(contour, 0)).float()


def load_class(path):

    cls0 = [1, 0, 0]
    cls1 = [0, 1, 0]
    cls2 = [0, 0, 1]
    if 'N' in os.path.basename(path):
        cls = cls0
    if 'D' in os.path.basename(path):
        cls = cls1
    if 'M' in os.path.basename(path):
        cls = cls2

    return torch.from_numpy(np.expand_dims(cls, 0)).long()


def load_distance(path, distance_type):

    if distance_type == "dist_mask":
        path = path.replace("image", "dis_mask").replace("png", "mat")
        # print (path)
        # print (io.loadmat(path))
        dist = io.loadmat(path)["dis"]

    if distance_type == "dist_contour":
        path = path.replace("image", "dis_contour").replace("png", "mat")
        dist = io.loadmat(path)["c_dis"]

    if distance_type == "dist_signed01":
        path = path.replace("image", "dis_signed01").replace("png", "mat")
        dist = io.loadmat(path)["s_dis01"]

    if distance_type == "dist_signed11":
        path = path.replace("image", "dis_signed11").replace("png", "mat")
        dist = io.loadmat(path)["s_dis11"]

    if distance_type == "dist_fore":
        path = path.replace("image", "dis_fore").replace("png", "mat")
        dist = io.loadmat(path)["f_dis"]

    return torch.from_numpy(np.expand_dims(dist, 0)).float()


class DatasetCornea(Dataset):

    def __init__(self, file_names, targetpaths):

        self.file_names = file_names
        self.targetpaths = targetpaths
        self.im_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(512),
            transforms.ColorJitter(0.2, 0.2, 0.0, 0.0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.label_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(512, interpolation=Image.NEAREST),
            # transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        name = os.path.split(self.file_names[idx])[1]
        img_file_name = os.path.splitext(name)[0]
        image = cv2.imread(self.file_names[idx])[..., ::-1]
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        _target = cv2.imread(self.targetpaths[idx])
        _target = cv2.resize(_target, (512, 512), interpolation=cv2.INTER_NEAREST)
        _target = (255 - _target)[..., 0] / 255.
        # _target[_target == 255] = 1

        im = Image.fromarray(np.uint8(image))
        target = Image.fromarray(np.uint8(_target)).convert('L')

        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            im_t = self.im_transform(im)

        torch.manual_seed(seed)
        random.seed(seed)
        if self.label_transform is not None:
            target_t = self.label_transform(target)
            # target_t = torch.from_numpy(np.asfarray(target_t).copy())
            target_t = torch.from_numpy(np.expand_dims(target_t, 0).copy()).float()

        # import imageio
        # im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t.permute(1, 2, 0).numpy()) * 255
        # imageio.imwrite('./debug/im.png', np.array(im_np).astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', np.array(target_np).astype(np.uint8))

        return img_file_name, im_t, target_t


class distancedStainingImage(Dataset):
    def __init__(self,
                 x,
                 y,
                 masks,
                 # names,
                 # args,
                 train=False):
        assert len(x) == len(y)
        assert len(x) == len(masks)
        # assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.masks = masks
        # self.names = names
        self.train = train

        # augmentation
        self.hflip = K.RandomHorizontalFlip()
        self.vflip = K.RandomVerticalFlip()
        self.jit = K.ColorJitter(0.2, 0.2, 0.05, 0.05)
        self.resize = kornia.geometry.resize
        self.normalize = K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]),
                                     std=torch.tensor([0.5, 0.5, 0.5]))
        # inductive bias
        # self.inductive_bias = args.inductive_bias

    def __len__(self):
        return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
            return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self._get_index(idx)

        # BGR -> RGB -> PIL
        image = cv2.imread(self.x[idx])[..., ::-1]
        label = cv2.imread(self.y[idx])
        label = (255 - label)[..., 0]
        mask = cv2.imread(self.masks[idx])
        mask = (255 - mask)[..., 0]
        name = os.path.split(self.x[idx])[1]

        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        embed_map = distance_transform_edt(mask)
        embed_map = (embed_map / np.max(embed_map) - 0.5) / 0.5
        embed_map = torch.tensor(embed_map)

        image_t = torch.tensor(image / 255).permute(2, 0, 1).unsqueeze(0)
        label_t = torch.tensor(label // 255).unsqueeze(0).float()
        map_t = embed_map.clone().detach().reshape(1, -1, image_t.size(-2), image_t.size(-1)).float()

        if self.train:
            hflip_params = self.hflip.forward_parameters(image_t.shape)
            image_t = self.hflip(image_t, hflip_params)
            label_t = self.hflip(label_t, hflip_params)
            map_t = self.hflip(map_t, hflip_params)
            vflip_params = self.vflip.forward_parameters(image_t.shape)
            image_t = self.vflip(image_t, vflip_params)
            label_t = self.vflip(label_t, vflip_params)
            map_t = self.vflip(map_t, vflip_params)
            image_t = self.resize(image_t, size=512, interpolation='bilinear', align_corners=False)
            label_t = self.resize(label_t, size=512, interpolation='nearest')
            map_t = self.resize(map_t, size=512, interpolation='nearest')
            jit_params = self.jit.forward_parameters(image_t.shape)
            image_t = self.jit(image_t, jit_params)
        else:
            image_t = self.resize(image_t, size=512, interpolation='bilinear', align_corners=False)
            label_t = self.resize(label_t, size=512, interpolation='nearest')
            map_t = self.resize(map_t, size=512, interpolation='nearest')
            map_t = map_t.view(1, -1, 512, 512)

        image_t = self.normalize(image_t).squeeze(0).float()
        label_t = label_t.long().squeeze(0)
        map_t = map_t.squeeze(0).float()

        # io debug
        # import imageio
        # im_np = image_t.permute(1, 2, 0).numpy()
        # im_np = (im_np * 0.5 + 0.5) * 255
        # gt_np = label_t.numpy() * 255
        # map_np = (map_t.squeeze().numpy() * 0.5 + 0.5) * 255
        # imageio.imwrite('./debug/im.png', im_np.astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', gt_np.astype(np.uint8))
        # imageio.imwrite('./debug/map.png', map_np.astype(np.uint8))

        # if self.inductive_bias != '':
        image_t = torch.cat([image_t, map_t], dim=0)

        return name, image_t, label_t
