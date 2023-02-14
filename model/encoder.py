''' Section 3.1 Multi-scale features from a pre-trained encoder of MuFAN in PyTorch.

Mainly adapted from https://github.com/autonomousvision/projected-gan/tree/main/pg_modules

Reference:
[1] Jung, Dahuin, et al. "New Insights for the Stability-Plasticity Dilemma in Online Continual Learning."
    International Conference on Learning Representations 2023.
'''

import torch
import torch.nn as nn
import timm
import torchvision


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, deconv=False, expand=False, align_corners=True):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        output = self.out_conv(output)

        return output


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], expand=expand)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0])

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch


def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def _make_detection_model(model):
    pretrained = nn.Module()

    pretrained.layer0 = nn.Sequential(*model.backbone.features[0][0:3])
    pretrained.layer1 = nn.Sequential(*model.backbone.features[0][3:5])
    pretrained.layer2 = nn.Sequential(*model.backbone.features[0][5:8])
    pretrained.layer3 = nn.Sequential(*model.backbone.features[0][8:12], nn.MaxPool2d(kernel_size=2, stride=2))

    return pretrained

def calc_channels(pretrained, inp_res=224):
    channels = []
    tmp = torch.zeros(1, 3, inp_res, inp_res)

    # forward pass
    tmp = pretrained.layer0(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer1(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer2(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer3(tmp)
    channels.append(tmp.shape[1])

    return channels


def _make_projector(im_res, cout, proj_type, expand=False, coco_ssdlite=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    if coco_ssdlite:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,
                                                                           pretrained_backbone=False)
        pretrained = _make_detection_model(model)
    else:
        model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
        pretrained = _make_efficientnet(model)

    # determine resolution of feature maps, this is later used to calculate the number
    # of down blocks in the discriminators. Interestingly, the best results are achieved
    # by fixing this to 128, ie., we use the same number of down blocks per discriminator
    # independent of the dataset resolution
    # im_res = 128
    pretrained.RESOLUTIONS = [im_res//2, im_res//4, im_res//8, im_res//16]
    pretrained.CHANNELS = calc_channels(pretrained)

    if proj_type == 0: return pretrained, None

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)
    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class Encoder_Proj(nn.Module):
    def __init__(
        self,
        im_res=128,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        coco_ssdlite=False
    ):
        super().__init__()
        self.proj_type = proj_type
        self.cout = cout
        self.expand = expand

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, cout=self.cout, proj_type=self.proj_type, expand=self.expand, coco_ssdlite=coco_ssdlite)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        # predict feature maps
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out
