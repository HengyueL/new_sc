import torch
from utils.model_parts import resnet34, LinearStandardized
from torchvision.models import resnet50 as rn50


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
            elif isinstance(m, torch.nn.Linear):
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
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)



if __name__ == "__main__":
    test_model = ResNet50Customized(num_classes=1000, dim_features=1024, standardized_linear_weights=False)
    test_input = torch.randn([1,3,224,224])
    test_output = test_model(test_input)
    print("Test Completed")