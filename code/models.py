import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class R34_ver1(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver1, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        ## freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        ## avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ## customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# Result:
# Loss: 3.912 | Acc: 2.000%


# more complex model + softmax + 5 crops
# epochs: 200
# batch_size: 32
class R34_ver2(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver2, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        ## freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        ## avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ## customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_cls),
            nn.Softmax(dim=1),  # Apply Softmax here for the final classification
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# No difference...


# Less complex model, adamw optimizer, 2 classifier layers, logsoftmax, added dropout
# epochs: 200
# batch_size: 32
class R34_ver3(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver3, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_cls),
            nn.LogSoftmax(dim=1),  # Apply LogSoftmax for the final classification
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# Reuslt:
# Time elapsed 0:00:14.841943
# [train-54/200] loss: 3.916924 | acc: 1.111%
# [val-54/200] loss: 3.912229 | acc: 2.000%
# Not Good


# much simple
# batch_size: 64
# learning_rate: 0.001
class R34_ver4(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver4, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(nn.Dropout(0.17), nn.Linear(512, num_cls))

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# Result:
# BIG improvement
# [train-100/100] loss: 0.298320 | acc: 91.244%
# [val-100/100] loss: 1.050834 | acc: 81.800%


# increase dropout rate to 0.23
# added one more linear layer
# added relu activation
# batch_size: 64
# use soft-voting
# epochs: 100
class R34_ver5(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver5, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.23),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


class R34_ver6(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver6, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.InstanceNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.17),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# avepool -> maxpool
class R34_ver7(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver7, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.InstanceNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.23),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.maxpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# smaller hidden layer, remove instance norm
class R34_ver9(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver9, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.17),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.17),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.maxpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# Result (Ensemble):
# [train-100/100] loss: 0.320171 | acc: 90.411%
# [val-100/100] loss: 0.974297 | acc: 76.400%
# Result (Single Model):
# [train-50/50] loss: 2.018832 | acc: 39.733%
# [val-50/50] loss: 1.881399 | acc: 43.000%


# gradually smaller hidden layer
class R34_ver8(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver8, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.17),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.17),
            nn.Linear(128, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.maxpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# [train-200/200] loss: 0.588439 | acc: 82.756%
# [val-200/200] loss: 1.566941 | acc: 66.600%


class R34_ver10(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver10, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.maxpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


class R34_ver11(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver11, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avepool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avepool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# adam optimizer
# epochs: 500
# batch_size: 64
# learning_rate: 0.001
class R34_ver12(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver12, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(nn.Dropout(0.22), nn.Linear(512, num_cls))

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


class R34_ver13(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver13, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# Use ELU activation function
class R34_ver14(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver14, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit


# Use ELU activation function
class R34_ver15(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver15, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            OrderedDict([*(list(resnet.named_children())[:-2])])
        )  # drop last layer which is classifier

        # Freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        # Avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Customized classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit
