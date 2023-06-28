import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import segmentation_models_pytorch as smp


# class Contrails_UNET(smp.Unet):
#     def __init__(self):
#         super(smp.Unet).__init__()
#         self.encoder_name = "timm-resnest26d",
#         self.encoder_weights = "imagenet",
#         self.in_channels = 3,
#         self.classes = 1,
#         self.activation = "sigmoid"

class Contrails_UNET(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.model = smp.Unet(
            # encoder_name="timm-efficientnet-b1", 
            encoder_name = "timm-resnest26d",
            encoder_weights="imagenet", 
            decoder_use_batchnorm=True,
            classes=1, 
            in_channels = 3,
            activation=None
        )
        self.loss_module = smp.losses.DiceLoss(mode="binary", smooth=1.0)
        #self.loss_module = smp.losses.DiceLoss(mode='binary')
    
    def forward(self, imgs):
        preds = self.model(imgs)
        return preds
        #return {"loss": loss, "logits": logits.sigmoid(), "logits_raw": logits, "target": y}

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CIFAR10Model(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CIFAR10_ResNet9(BaseModel):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)