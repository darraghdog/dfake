import os
import torch
import torch.nn as nn
try:
    from .utils import ShuffleV1Block
except ImportError:
    from utils import ShuffleV1Block


class ShuffleNetV1(nn.Module):
    def __init__(self,
                 input_size=224,
                 num_classes=1000,
                 width_mult=0.5,
                 group=8,
                 return_type='before_pool',
                 pretrained=True):
        super(ShuffleNetV1, self).__init__()
        assert group is not None

        self.stage_repeats = [4, 8, 4]
        self.width_mult = width_mult
        if group == 3:
            if width_mult == 0.5:
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif width_mult == 1.0:
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif width_mult == 1.5:
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif width_mult == 2.0:
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if width_mult == 0.5:
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif width_mult == 1.0:
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif width_mult == 1.5:
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif width_mult == 2.0:
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        RETURN_TYPES = {'before_last_conv': 0, 'before_pool': 1, 'before_classifier': 2, 'after_classifier': 3}

        if return_type not in RETURN_TYPES:
            raise ValueError(
                "Wrong retyrn_type ! Use one of : {before_last_conv, before_pool, before_classifier, after_classifier}")
        self.return_type = RETURN_TYPES[return_type]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                self.features.append(
                    ShuffleV1Block(input_channel,
                                   output_channel,
                                   group=group,
                                   first_group=first_group,
                                   mid_channels=output_channel // 4,
                                   ksize=3,
                                   stride=stride))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        # building last several layers
        if self.return_type == 0:  # if type > before_last_conv
            print('WARNING : no last_conv layer !')
        if self.return_type > 1:  # 'before_classifier' and 'after_classifier'
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.return_type == 3:
            self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], num_classes, bias=False))

        if not pretrained:
            self._initialize_weights()
        else:
            curr_dir = os.path.dirname(__file__)
            weights_path = f'{curr_dir}/weights/shufflenet_v1_weights/group{group}_{width_mult:.1f}x.pth.tar'
            print('load pretrained weights from', weights_path)
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        if self.return_type < 2:
            return x

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if self.return_type == 2:
            return x

        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    group = 3
    width_mult = 0.5
    model = ShuffleNetV1(group=group, width_mult=width_mult, return_type='after_classifier')
    model.eval()

    test_data = torch.zeros(5, 3, 224, 224)
    with torch.no_grad():
        test_outputs = model(test_data)
    print(test_outputs[0])

    from thop import profile, clever_format
    from ptflops import get_model_complexity_info

    with torch.no_grad():
        print('FLOPS by thop (Lyken17) :')
        flops, params = profile(model, inputs=(torch.zeros(1, 3, 224, 224), ), verbose=False)
        flops, params = clever_format([flops, params], "%.5f")
        print('Flops:  ' + flops)
        print('Params: ' + params)
