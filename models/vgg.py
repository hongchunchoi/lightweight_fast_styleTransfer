import torchvision






def VGG16() :
    vgg16 = torchvision.models.vgg16_bn(pretrained=True)
