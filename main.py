import numpy as np
import torch
import torchvision.transforms as transforms
from models.vgg import *
from models.transformer import  *
import torchvision.datasets as datasets
from utils import *

random_seed = 1994
batch_size = 128
device = ""

np.random.seed(random_seed)
torch.manual_seed(random_seed)
initial_lr = 0.1

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
])


train_dataset = datasets.ImageFolder("C:/Users/Hongjun/Desktop/dataset/coco", transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

transformer = TransformerNet()
vgg = VGG16(requires_grad=False).to(device)

optimizer = torch.optim.Adam(transformer.parameters(), initial_lr)
mse_loss = nn.MSELoss()

style = load_image(filename="./styleImage/", size=None, scale=None)

style = style_transform(style)
style = style.repeat(batch_size, 1, 1, 1).to(device)

features_style = vgg(normalize_batch(style))
gram_style = [gram_matrix(y) for y in features_style]