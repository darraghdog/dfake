import sys
import os
import torch
import torch.nn as nn
from shufflenet import Shufflenet, Shuffle_Xception, HS, SELayer



class ShuffleNetV2_Plus(nn.Module):
    def __init__(
        self,
        input_size=224,
        device='cuda',
        model_size='Large',
        pretrained=True,
        return_type='before_pool',
        num_classes=1000,
    ):
        super(ShuffleNetV2_Plus, self).__init__()

        assert input_size % 32 == 0
        architecture = (0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2)

        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
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
            HS(),
        )

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    #print('Shuffle3x3')
                    self.features.append(
                        Shufflenet(inp,
                                   outp,
                                   base_mid_channels=outp // 2,
                                   ksize=3,
                                   stride=stride,
                                   activation=activation,
                                   useSE=useSE))
                elif blockIndex == 1:
                    #print('Shuffle5x5')
                    self.features.append(
                        Shufflenet(inp,
                                   outp,
                                   base_mid_channels=outp // 2,
                                   ksize=5,
                                   stride=stride,
                                   activation=activation,
                                   useSE=useSE))
                elif blockIndex == 2:
                    #print('Shuffle7x7')
                    self.features.append(
                        Shufflenet(inp,
                                   outp,
                                   base_mid_channels=outp // 2,
                                   ksize=7,
                                   stride=stride,
                                   activation=activation,
                                   useSE=useSE))
                elif blockIndex == 3:
                    #print('Xception')
                    self.features.append(
                        Shuffle_Xception(inp,
                                         outp,
                                         base_mid_channels=outp // 2,
                                         stride=stride,
                                         activation=activation,
                                         useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)
        self.features = nn.Sequential(*self.features)
        self.head_input_dim = input_channel

        if self.return_type > 0:  # if type > before_last_conv
            self.conv_last = nn.Sequential(nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False), nn.BatchNorm2d(1280),
                                           HS())
            self.head_input_dim = 1280
        if self.return_type > 1:  # 'before_classifier' and 'after_classifier'
            self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
            self.LastSE = SELayer(1280)

        if self.return_type == 3:
            self.fc = nn.Sequential(
                nn.Linear(1280, 1280, bias=False),
                HS(),
            )
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Sequential(nn.Linear(1280, num_classes, bias=False))
            self.head_input_dim = num_classes

        if not pretrained:
            self._initialize_weights()
        else:
            curr_dir = os.path.dirname(__file__)
            weights_path = f'{curr_dir}/../weights/shufflenet/ShuffleNetV2+.{model_size}.pth.tar'
            print('load pretrained weights from', weights_path)
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict, strict=False, map_location=device)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        if self.return_type == 0:
            return x

        x = self.conv_last(x)
        if self.return_type == 1:
            return x

        x = self.globalpool(x)
        x = self.LastSE(x)
        x = x.contiguous().view(-1, 1280)

        if self.return_type == 2:
            return x

        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
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
    model_size = 'Large'
    model = ShuffleNetV2_Plus(model_size=model_size, return_type='after_classifier')
    model.eval()

    # print(model)

    test_data = torch.zeros(5, 3, 224, 224)
    with torch.no_grad():
        test_outputs = model(test_data)
    print(test_outputs[0][0:10])

    # from thop import profile, clever_format
    # from ptflops import get_model_complexity_info

    # with torch.no_grad():
    #     print('FLOPS by thop (Lyken17) :')
    #     flops, params = profile(model, inputs=(torch.zeros(1, 3, 224, 224), ), verbose=False)
    #     flops, params = clever_format([flops, params], "%.5f")
    #     print('Flops:  ' + flops)