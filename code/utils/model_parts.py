import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==== ResNet Backbones (feature encoder) ====
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, base_width=64):
        super(ResNet, self).__init__()
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        print("Last linear shape: ", base_width*8*block.expansion)
        # self.linear = nn.Linear(base_width*8*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # print("Check Shape")
        x = torch.flatten(x,1)
        # out = self.linear(out)
        return out

def resnet18(base_width=64):
    return ResNet(BasicBlock, [2,2,2,2], base_width=base_width)

def resnet34(base_width=64):
    return ResNet(BasicBlock, [3,4,6,3], base_width=base_width)


# === Standardized Linear Layer ===
class LinearStandardized(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # If use bias, things gonna change a bit when computing the forward path
        self.use_bias = bias  
       
    def forward(self, input):
        if self.use_bias:
            new_weight = torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)
            new_input = torch.cat([input, torch.ones([input.shape[0], 1])], dim=1)
        else:
            new_input = input
            new_weight = self.weight
            
        new_bias = None
        scale = torch.linalg.norm(new_weight, dim=1, keepdim=True)
        new_weight = new_weight / scale
        return F.linear(new_input, new_weight, new_bias)
    

if __name__ == "__main__":

    # == Test if customized fc works well ==
    test_layer = LinearStandardized(
        in_features=20,
        out_features=5,
        bias=True
    )
    loss_func = nn.SmoothL1Loss()

    test_input = torch.randn([2, 20]).to(dtype=torch.float)
    test_label = torch.ones([2, 5]).to(dtype=torch.float)

    test_output = test_layer(test_input)
    loss = loss_func(test_output, test_label)
    loss.backward()
    
    print("Check if grad successfully passed here [weight]: ", test_layer.weight.grad.shape)
    print("Check if grad successfully passed here [bias]: ", test_layer.bias.grad.shape)
    # === test passed ===