import torch
from utils.model_parts import resnet34, LinearStandardized
from torchvision.models import resnet50 as rn50
import timm
TIMM_MODEL_CARDS = {
    "Timm-ResNext": "seresnextaa101d_32x8d.sw_in12k_ft_in1k",
    "Timm-WideResnet": "wide_resnet101_2.tv2_in1k",
    "Timm-VOLO": "volo_d4_224.sail_in1k",
    "Timm-ConvNext": "convnextv2_base.fcmae_ft_in22k_in1k",

}

class ResNet34Customized(torch.nn.Module):
    """
        Customized resnet with option to set the fc layer to have standardized weights 
    """
    def __init__(
            self, num_classes=10, dim_features=512, init_weights=True, 
            standardized_linear_weights=False
        ):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
            standardized_linear_weights:  if use classification use w * x /||w|| + b 
        """
        super(ResNet34Customized, self).__init__()
        self.features = resnet34()
        self.dim_features = dim_features
        self.num_classes = num_classes
        
        # represented as f() in the original paper
        if not standardized_linear_weights:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.dim_features, self.num_classes)
            )
        else:
            self.classifier = torch.nn.Sequential(
                LinearStandardized(self.dim_features, self.num_classes)
            )
        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.features)
            self._initialize_weights(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        prediction_out = self.classifier(x)
        return prediction_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear) or isinstance(m, LinearStandardized):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


class ResNet50Customized(torch.nn.Module):
    """
        Customized resnet with option to set the fc layer to have standardized weights 
    """
    def __init__(
            self, num_classes=10, dim_features=2048, init_weights=True, 
            standardized_linear_weights=False
        ):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
            standardized_linear_weights:  if use classification use w * x /||w|| + b 
        """
        super(ResNet50Customized, self).__init__()

        weights = None if init_weights else "pretrained"
        self.features = rn50(weights=weights)
        self.dim_features = 2048
        self.num_classes = num_classes
        
        # This changes the torch default resnet structure minimally.
        if not standardized_linear_weights:
            pass
        else:
            self.classifier = LinearStandardized(self.dim_features, self.num_classes)
            self.features.fc = self.classifier

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.features)
        
    def forward(self, x):
        prediction_out = self.features(x)
        return prediction_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear) or isinstance(m, LinearStandardized):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def build_timm_model(model_name, standardized_linear_weights=False):
    model_card_name = TIMM_MODEL_CARDS[model_name]
    model = timm.create_model(
        model_card_name, pretrained=True
    )
    if model_name in ["Timm-ConvNext", "Timm-CoAtNet", "Timm-mViT"]:
        n_in, n_out = model.head.fc.in_features, model.head.fc.out_features
        if standardized_linear_weights:
            model.head.fc = LinearStandardized(n_in, n_out)
        else:
            model.head.fc = torch.nn.Linear(n_in, n_out)
        torch.nn.init.normal_(model.head.fc.weight, 0, 0.01)
        torch.nn.init.constant_(model.head.fc.bias, 0)
    elif model_name in ["Timm-VOLO", "Timm-MLP", "Timm-DeiT", "Timm-EVA"]:
        n_in, n_out = model.head.in_features, model.head.out_features
        if standardized_linear_weights:
            model.head = LinearStandardized(n_in, n_out)
        else:
            model.head = torch.nn.Linear(n_in, n_out)
        torch.nn.init.normal_(model.head.weight, 0, 0.01)
        torch.nn.init.constant_(model.head.bias, 0)
    elif model_name in ["Timm-WideResNet", "Timm-ResNext"]:
        n_in, n_out = model.fc.in_features, model.fc.out_features
        if standardized_linear_weights:
            model.fc = LinearStandardized(n_in, n_out)
        else:
            model.fc = torch.nn.Linear(n_in, n_out)
        torch.nn.init.normal_(model.fc.weight, 0, 0.01)
        torch.nn.init.constant_(model.fc.bias, 0)
    elif model_name in ["Timm-MobileNet", "Timm-EfficientNet"]:
        n_in, n_out = model.classifier.in_features, model.classifier.out_features
        if standardized_linear_weights:
            model.classifier = LinearStandardized(n_in, n_out)
        else:
            model.classifier = torch.nn.Linear(n_in, n_out)
        torch.nn.init.normal_(model.classifier.weight, 0, 0.01)
        torch.nn.init.constant_(model.classifier.bias, 0)
    else:
        raise RuntimeError("Unimplemented Timm model creation.")
    return model


def build_dino_model(standardized_linear_weights=False, use_pretrained=False):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
    if not use_pretrained:
        print("Random Init a New Dino Fc Head.")
        fc = model.linear_head
        in_f, out_f = fc.in_features, fc.out_features
        if standardized_linear_weights:
            model.linear_head = LinearStandardized(in_f, out_f)
        else:
            model.linear_head = torch.nn.Linear(in_f, out_f)
        # torch.nn.init.normal_(model.linear_head.weight)
        torch.nn.init.constant_(model.linear_head.weight, 0.01)
        torch.nn.init.constant_(model.linear_head.bias, 0)
    return model

if __name__ == "__main__":
    # test_model = ResNet50Customized(num_classes=1000, dim_features=1024, standardized_linear_weights=False)
    # test_input = torch.randn([1,3,224,224])
    # test_output = test_model(test_input)
    # print("Test Completed")

    test_model = build_timm_model("Timm-ResNext", True)
    test_input = torch.randn([1,3,224,224])
    test_output = test_model(test_input)
    print("Test Completed")