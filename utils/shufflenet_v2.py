import os
import torch
import torch.nn as nn
try:
    from .utils import ShuffleV2Block
except ImportError:
    from utils import ShuffleV2Block


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, width_mult=0.5, return_type='after_classifier', pretrained=True):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.width_mult = width_mult
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
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
                if i == 0:
                    self.features.append(
                        ShuffleV2Block(input_channel,
                                       output_channel,
                                       mid_channels=output_channel // 2,
                                       ksize=3,
                                       stride=2))
                else:
                    self.features.append(
                        ShuffleV2Block(input_channel // 2,
                                       output_channel,
                                       mid_channels=output_channel // 2,
                                       ksize=3,
                                       stride=1))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        if self.return_type > 0:  # if type > before_last_conv
            output_channel = 1024 if self.width_mult != 2.0 else 2048
            self.conv_last = nn.Sequential(nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True))
        if self.return_type > 1:  # 'before_classifier' and 'after_classifier'
            self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
            if self.width_mult == 2.0:
                self.dropout = nn.Dropout(0.2)
        if self.return_type == 3:
            self.classifier = nn.Sequential(nn.Linear(output_channel, n_class, bias=False))

        if not pretrained:
            self._initialize_weights()
        else:
            curr_dir = os.path.dirname(__file__)
            weights_path = f'{curr_dir}/weights/shufflenet_v2_weights/ShuffleNetV2.{width_mult:.1f}x.pth.tar'
            print('load pretrained weights from', weights_path)
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        if self.return_type == 0:
            return x

        x = self.conv_last(x)
        if self.return_type == 1:
            return x

        x = self.globalpool(x)
        if self.width_mult == 2.0:
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])

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


def check_networks():

    for i in range(3):
        t = torch.rand(1, 3, 224, 224)
        for width_mult in [0.5, 1.0, 1.5, 2.0]:
            model1 = ShuffleNetV2(width_mult=width_mult, return_type='after_classifier')
            model1.eval()

            model2 = ShuffleNetV2(width_mult=width_mult, return_type='before_classifier')
            model2.eval()
            model1.return_type = 2
            with torch.no_grad():
                out1 = model1(t)
                out2 = model2(t)
                assert (torch.allclose(out1, out2))


# if __name__ == "__main__":
#     check_networks()

if __name__ == "__main__":
    width_mult = 2.0
    model = ShuffleNetV2(width_mult=width_mult, return_type='after_classifier')
    model.eval()

    test_data = torch.zeros(8, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs[:][0:10])

    from thop import profile, clever_format
    from ptflops import get_model_complexity_info

    with torch.no_grad():
        print('FLOPS by thop (Lyken17) :')
        flops, params = profile(model, inputs=(torch.zeros(1, 3, 224, 224), ), verbose=False)
        flops, params = clever_format([flops, params], "%.5f")
        print('Flops:  ' + flops)
        print('Params: ' + params)

# torch.onnx.export(model, torch.zeros(1, 3, 224, 224), '/home/vladislav.leketush/Videos/test_model.onnx')
