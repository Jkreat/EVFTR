import torch
from .DAEMA import DenseASPPWithEMAHead
from .ASPP import AtrousSpatialPyramidPoolingHead as ASPPHead
from . import resnet
from torchvision.models._utils import IntermediateLayerGetter

def createDenseASPPWithEMAHead(in_channels, classes):
    return DenseASPPWithEMAHead(in_channels, classes)

def createASPPHead(in_channels, classes):
    return ASPPHead(in_channels, classes)

def createDecoder(decoder_name, in_channels, classes):
    if decoder_name == 'DAEMA':
        return createDenseASPPWithEMAHead(in_channels, classes)
    elif decoder_name == 'ASPP':
        return createASPPHead(in_channels, classes)

class EVFRNet_lite(torch.nn.Module):
    def __init__(self, stage,
                 decoder1,
                 decoder2,
                 ckpt_path = None):
        super(EVFRNet_lite, self).__init__()
        self.ckpt_path = ckpt_path
        self.stage = stage
        self.decoder1_name = decoder1
        self.decoder2_name = decoder2
        self.ignore_label = None

        if not self.stage in [0, 1, 2]:
            raise ValueError('Stage must be 0, 1 or 2!')
        '''Stage
        0: Train 4-Class semantic segmentation only
        1: Train BG-Extracting network first
        2: After stage1, load its weights and freeze backbone
        '''

        backbone = resnet.__dict__['resnet50'](
            pretrained=False,
            replace_stride_with_dilation=[False, True, True])
        backbone = IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})
        self.backbone = backbone

        if stage==0:
            self._init_stage0()
        elif stage==1:
            self._init_stage1()
        elif stage==2:
            self._init_stage2()

    def _init_stage0(self):
        num_classes = 4
        self.decoder1 = None
        self.decoder2 = createDecoder(self.decoder2_name, 2048, num_classes)
        pass

    def _init_stage1(self):
        num_classes = 2
        self.decoder1 = createDecoder(self.decoder1_name, 2048, num_classes)
        self.decoder2 = None
        # Todo: Need to freeze some backbone parameters here
        # backbone.layerx.eval()

    def _init_stage2(self):
        self.decoder1 = createDecoder(self.decoder1_name, 2048, 2)
        self.decoder2 = createDecoder(self.decoder2_name, 2048, 4)

        self.backbone.eval()
        self.decoder1.eval()

    def _init_eval(self):
        # todo: set eval mode out of this class may be better?
        pass

    def _forward_resnet_layers(self, x):
        # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (relu): ReLU(inplace=True)
        #   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        return x1, x2, x3, x4

    def _forward_stage0(self, x):
        if self.decoder2_name in ['OCR', 'ASPOCR']:
            x = self._forward_ocr(x)
        else:
            x = self.backbone(x)['out']
            if self.decoder2_name == 'DenseASPPWithEMA':
                x, mus = self.decoder2(x)
            else:
                x = self.decoder2(x)
        x_T = torch.nn.functional.interpolate(x, (420, 560),
                                              mode='bilinear',
                                              align_corners=False)
        return x_T

    def _forward_stage1(self, x):
        x = self.backbone(x)['out']
        x = self.decoder1(x)
        x_T = torch.nn.functional.interpolate(x, (420, 560),
                                              mode='bilinear',
                                              align_corners=False)
        return x_T

    def _forward_stage2(self, x):
        if self.decoder2_name in ['OCR', 'ASPOCR']:
            _, _, x3, x4 = self._forward_resnet_layers(x)
            x_res1 = self.decoder1(x4)
            _, x_res2 = self.decoder2(x3, x4)
        else:
            x = self.backbone(x)['out']
            x_res1 = self.decoder1(x)
            if self.decoder2_name == 'DAEMA':
                x_res2, _ = self.decoder2(x)
            else:
                x_res2 = self.decoder2(x)

        # fixme: really need detach()?

        # x_res1: batch, 2, H, W
        # x_res2: batch, 4, H, W, ignore_label = 0?

        x_res1 = torch.nn.functional.interpolate(x_res1, (420, 560),
                                                 mode='bilinear',
                                                 align_corners=False)
        x_res2 = torch.nn.functional.interpolate(x_res2, (420, 560),
                                                 mode='bilinear',
                                                 align_corners=False)

        x_res1_map = torch.argmin(x_res1, dim=1) * 100  # batch, 1, H, W
        x_res1_map.float()
        x_res_all = x_res2.clone()
        x_res_all[:, 0, :, :] = x_res1_map

        return {'res1': x_res1, 'res2': x_res2, 'res_all': x_res_all}

    def forward(self, x):
        if self.stage==0:
            return self._forward_stage0(x)
        elif self.stage==1:
            return self._forward_stage1(x)
        elif self.stage==2:
            return self._forward_stage2(x)
