import torch
import torch.nn as nn

import timm

class CustomCNN(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,padding=3,stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc= nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        logits = self.fc(x)
        
        return logits
    
class multispectralTimmModelWrapper(nn.Module):
    def __init__(self, arch_name, num_channels, num_classes, pretrained):
        super(multispectralTimmModelWrapper, self).__init__()

        #self.arch_name=arch_name
        #self.num_channels=num_channels
        #self.num_classes=num_classes
        #self.pretrained=pretrained

        if arch_name == "ResNet18":
            #https://huggingface.co/timm/resnet18.a1_in1k
            self.model = timm.create_model("resnet18.a1_in1k", pretrained=pretrained, in_chans=num_channels, num_classes=num_classes)
            
        if arch_name == "ConvNeXt-Nano":
            #https://huggingface.co/timm/convnext_nano.in12k
            self.model = timm.create_model("convnext_nano.in12k", pretrained=pretrained, in_chans=num_channels, num_classes=num_classes)

        if arch_name == "ViT-Tiny":
            #https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k
            self.model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=pretrained, in_chans=num_channels, img_size=112, patch_size=8, num_classes=num_classes)

        #data_config = timm.data.resolve_model_data_config(model)                    #TODO check if necessary
        #transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self,x):
        return self.model(x)


def get_network(arch_name, num_channels, num_classes, pretrained):
    if arch_name == "CustomCNN":
        model = CustomCNN(num_channels, num_classes)
    else:
        model = multispectralTimmModelWrapper(arch_name, num_channels, num_classes, pretrained)
    
    #model = model.eval() not necessary since lightning sets mode automatically
    return model


if __name__ == "__main__":
    for arch_name in ["ResNet18","ConvNeXt-Nano","ViT-Tiny"]:
        network = get_network(arch_name=arch_name,num_channels=10,num_classes=19,pretrained=True)
        print(arch_name, dict(network.model.named_children()).keys())
        print(list(network.model.named_children())[0])
        print(list(network.model.named_children())[-1],"\n\n")