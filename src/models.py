import torchvision.models as models
import torch.nn as nn

def get_model(model_name, classes = 10):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
    
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 10)
    
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 10)
    
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(2048, 10)
    
    return model

def get_embedding_model(model_name):
    if model_name == "resnet18":
        model = ResNet18E()
    
    elif model_name == "resnet50":
        model = ResNet50E()
    
    elif model_name == "densenet121":
        model = DenseNet121E()

    elif model_name == "wide_resnet50_2":
        model = WideResNet50V2E()
    
    return model


class ResNet18E(nn.Module):
    def __init__(self):
        super(ResNet18E, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)
        
    def forward(self,x):
        x = self.model(x)
        return x
    
    def embedding_first(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x.reshape(x.shape[0], -1)
    
    def embedding_layer1(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer2(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer3(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer4(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x.reshape(x.shape[0], -1)

class ResNet50E(nn.Module):
    def __init__(self):
        super(ResNet50E, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 10)
        
    def forward(self,x):
        x = self.model(x)
        return x
    
    def embedding_first(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x.reshape(x.shape[0], -1)
    
    def embedding_layer1(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer2(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer3(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer4(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x.reshape(x.shape[0], -1)

class DenseNet121E(nn.Module):
    def __init__(self):
        super(DenseNet121E, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, 10)
        
    def forward(self,x):
        x = self.model(x)
        return x

    def embedding(self, x):
        x = self.model.features(x)
        return x.reshape(x.shape[0], -1)


class WideResNet50V2E(nn.Module):
    def __init__(self):
        super(WideResNet50V2E, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True)
        self.model.fc = nn.Linear(2048, 10)
        
    def forward(self,x):
        x = self.model(x)
        return x
    
    def embedding_first(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x.reshape(x.shape[0], -1)
    
    def embedding_layer1(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer2(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer3(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x.reshape(x.shape[0], -1)

    def embedding_layer4(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x.reshape(x.shape[0], -1)