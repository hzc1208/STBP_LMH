from unicodedata import numeric
import torch.nn as nn
import torch

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ],
    'vggdvs': [
        [64, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A'],
        []
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )         
        elif num_classes == 200:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*2*2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )            
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out

class VGG_nobias(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_nobias, self).__init__()
        self.init_channels = 3
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out


class VGG_normed(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_normed, self).__init__()
        self.num_classes = num_classes
        self.module_list = self._make_layers(cfg[vgg_name], dropout)


    def _make_layers(self, cfg, dropout):
        layers = []
        for i in range(5):
            for x in cfg[i]:
                if x == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(3, x, kernel_size=3, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    self.init_channels = x
        layers.append(nn.Flatten())
        if self.num_classes == 1000:
            layers.append(nn.Linear(512*7*7, 4096))
        else:
            layers.append(nn.Linear(512, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module_list(x)


class VGGDVS(nn.Module):
    def __init__(self, vgg_name, T, num_class, init_c=2, init_s=48):
        super(VGGDVS, self).__init__()
        self.T = T
        self.init_channels = init_c
        
        cnt = 0
        for l in cfg[vgg_name]:
            if len(l)>0:
                cnt += 1
        
        self.W = int(init_s/(1<<cnt))**2

        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.classifier = self._make_classifier(num_class, cfg[vgg_name][cnt-1][1])
        
        #self.merge = MergeTemporalDim(T)
        #self.expand = ExpandTemporalDim(T)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class, channels):
        #layer = [nn.Linear(channels*self.W, num_class)]
        layer = [nn.Linear(channels*self.W, 100), nn.AvgPool1d(10, 10)]
        #layer = [nn.Dropout(0.25), nn.Linear(channels*self.W, 512), nn.Dropout(0.25), nn.Linear(512, num_class)]
        return nn.Sequential(*layer)

    def forward(self, input):
        #input = self.merge(input)
        
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        #out = self.expand(out)
        return out


def vgg11(num_classes=10, dropout=0, **kargs):
    return VGG('VGG11', num_classes, dropout)


def vgg13(num_classes=10, dropout=0, **kargs):
    return VGG('VGG13', num_classes, dropout)


def vgg16(num_classes=10, dropout=0, bias=True, **kargs):
    if bias == False:
        return VGG_nobias('VGG16', num_classes, dropout)
    else:
        return VGG('VGG16', num_classes, dropout)


def vgg19(num_classes=10, dropout=0, **kargs):
    return VGG('VGG19', num_classes, dropout)


def vgg16_normed(num_classes=10, dropout=0, **kargs):
    return VGG_normed('VGG16', num_classes, dropout)
    
    
def vggdvs(num_classes=10, T=10, **kargs):
    return VGGDVS('vggdvs', T, num_classes)
    
    
class VotingLayer(nn.Module):
    def __init__(self, voting_size: int = 10):
        super().__init__()
        self.voting_size = voting_size

    def forward(self, x: torch.Tensor):
        y = nn.functional.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)
        return y

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes, channels=256):
        super().__init__()

        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sn1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sn2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.sn3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(channels)
        self.sn4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(channels)
        self.sn5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(channels)
        self.sn6 = nn.ReLU(inplace=True)

        self.pool6 = nn.MaxPool2d(2, 2)

        self.dp7 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(channels * 8 * 8, 2048)
        self.sn7 = nn.ReLU(inplace=True)

        self.dp8 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(2048, 100)
        self.sn8 = nn.ReLU(inplace=True)
        self.voting = VotingLayer(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sn4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        x = self.pool6(x)

        x = x.flatten(1)

        x = self.dp7(x)
        x = self.fc7(x)
        x = self.sn7(x)

        x = self.dp8(x)
        x = self.fc8(x)
        x = self.sn8(x)

        x = self.voting(x)
        return x