import copy
from .resnet import ResNet
from .ghostnet import GhostNet
from .shufflenetv2 import ShuffleNetV2
from .mobilenetv2 import MobileNetV2
from .efficientnet_lite import EfficientNetLite
from .custom_csp import CustomCspNet
from .repvgg import RepVGG
from .mobilenetv3 import MobileNetV3_Small
from .mobilenetv2_s import MobileNetV2S
from .shufflenetv2_s import ShuffleNetV2S
from .mobilenetv2_s2 import MobileNetV2S2
from .mobilenetv3_2 import MobileNetV3


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'ResNet':
        return ResNet(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    elif name == 'GhostNet':
        return GhostNet(**backbone_cfg)
    elif name == 'MobileNetV2':
        return MobileNetV2(**backbone_cfg)
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'CustomCspNet':
        return CustomCspNet(**backbone_cfg)
    elif name == 'RepVGG':
        return RepVGG(**backbone_cfg)
    elif name == 'MobileNetV2S':
        return MobileNetV2S(**backbone_cfg)
    elif name == 'ShuffleNetV2S':
        return ShuffleNetV2S(**backbone_cfg)
    elif name == 'MobileNetV2S2':
        return MobileNetV2S2(**backbone_cfg)
    elif name == 'MobileNetV3':
        return MobileNetV3_Small(**backbone_cfg)
    else:
        raise NotImplementedError

