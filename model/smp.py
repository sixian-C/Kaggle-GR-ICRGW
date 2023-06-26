
import segmentation_models_pytorch as smp

model = smp.Unet(
        encoder_name ='timm-resnest26d',
        encoder_weights='imagenet',    # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,        # model output channels (number of classes in your dataset)
        activation="sigmoid",
        )

class uunet(smp.Unet):
    def __init__(self, num_classes=10):
        super().__init__()
print(uunet)