from typing import Any, cast, Dict, List, Union

__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights",
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]

class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def VGG_11(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["A"], batch_norm=False), **kwargs)
    return model
def VGG_11_BN(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["A"], batch_norm=True), **kwargs)
    return model
def VGG_13(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["B"], batch_norm=False), **kwargs)
    return model
def VGG_13_BN(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["B"], batch_norm=True), **kwargs)
    return model
def VGG_16(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["D"], batch_norm=False), **kwargs)
    return model
def VGG_16_BN(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["D"], batch_norm=True), **kwargs)
    return model
def VGG_19(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["E"], batch_norm=False), **kwargs)
    return model
def VGG_19_BN(**kwargs) -> VGG:
    model = VGG(make_layers(cfgs["E"], batch_norm=True), **kwargs)
    return model

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],}