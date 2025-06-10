import torch.utils.data as data
import PIL.Image as Image
import glob
from torchvision import transforms


x_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])

y_transforms = transforms.ToTensor()

def make_dataset(path, dn):
    imgs = []
    for img in glob.glob(f'/ho/data/{dn}/' + path + '/img/'
                         + '*.png'):
        mask = img.replace('img', 'mask')
        imgs.append((img, mask))
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, mode, dn):
        imgs = make_dataset(mode, dn)
        self.imgs = imgs

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('RGB')

        img_x = x_transforms(img_x)
        img_y = y_transforms(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
