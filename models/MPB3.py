import torch
import torch.nn as nn
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnetsp18, resnetsp34, resnetsp50
import torchvision
from models.utilities import SEBasicBlock, SELayer, CBAM

class MPB3net(nn.Module):

    def __init__(self, backbone='resnet18', r=16, pretrained=False, n_class=9, n_units=[128, 128], output_form='dual'):
        super(MPB3net, self).__init__()

        if 'resnet18' in backbone:
            feature_extractor = resnet18(pretrained=pretrained)
        elif 'resnetsp18' in backbone:
            feature_extractor = resnetsp18()
        elif 'resnetsp34' in backbone:
            feature_extractor = resnetsp34()
        elif 'resnetsp50' in backbone:
            feature_extractor = resnetsp50()
        elif 'resnet34' in backbone:
            feature_extractor = resnet34(pretrained=pretrained)
        elif 'resnet50' in backbone:
            feature_extractor = resnet50(pretrained=pretrained)
        elif 'resnet101' in backbone:
            feature_extractor = resnet101(pretrained=pretrained)

        elif 'mobilenetv3small' in backbone:
            feature_extractor = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        elif 'mobilenetv3large' in backbone:
            feature_extractor = torchvision.models.mobilenet_v3_large(pretrained=pretrained, width_mult=1.0,  reduced_tail=False, dilated=False)
        elif 'mobilenetv3quant' in backbone:
            feature_extractor = torchvision.models.quantization.mobilenet_v3_large(pretrained=pretrained)

        elif 'efficientnetb0' in backbone: # memory consumption issue not fixed yet
            feature_extractor = torchvision.models.efficientnet_b0(pretrained=pretrained)
        elif 'efficientnetb1' in backbone:
            feature_extractor = torchvision.models.efficientnet_b1(pretrained=pretrained)
        elif 'efficientnetb2' in backbone:
            feature_extractor = torchvision.models.efficientnet_b2(pretrained=pretrained)
        else:
            print('not implemented')

        self.output_form = output_form
        bos_fc_out = [n_units[0], 2]
        bom_fc_out = [n_units[1], n_class]
        if self.output_form == 'dual':
            bos_c_multiplier = 2
            bom_c_multiplier = 1
        elif self.output_form == 'dual2':
            bos_c_multiplier = 2
            bom_c_multiplier = 2
        elif self.output_form == 'mclass':
            bos_c_multiplier = 2
            bom_c_multiplier = 2
        else:
            print('Not implemented')
            assert False
        if backbone.startswith('sepost'):
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             # feature_extractor.avgpool
                                             )

                feature_channel = self.cnn_encoder[-1][-1].conv1.weight.shape[1]
            elif 'mobilenet' in backbone or 'efficientnet' in backbone:
                self.cnn_encoder = feature_extractor.features
                feature_channel = feature_extractor.features[-1].out_channels

            se_attention_bos = SEBasicBlock(int(feature_channel * bos_c_multiplier), int(feature_channel * bos_c_multiplier), reduction=r)
            se_attention_bom = SEBasicBlock(int(feature_channel * bom_c_multiplier), int(feature_channel * bom_c_multiplier), reduction=r)

            # branch of similarity
            self.head_bos = nn.Sequential(
                se_attention_bos,
                nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                nn.Flatten(),
                nn.BatchNorm1d(int(feature_channel * bos_c_multiplier)),
                nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                nn.BatchNorm1d(bos_fc_out[0]),
                nn.ReLU(inplace=True),
                nn.Linear(bos_fc_out[0], bos_fc_out[1])
                # nn.Sigmoid()
            )
            # branch of multi-classification
            self.head_bom = nn.Sequential(
                se_attention_bom,
                nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                nn.Flatten(),
                nn.BatchNorm1d(int(feature_channel * bom_c_multiplier)),
                nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                nn.BatchNorm1d(bom_fc_out[0]),
                nn.ReLU(inplace=True),
                nn.Linear(bom_fc_out[0], bom_fc_out[1]))

        elif backbone.startswith('fcdropout'):
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4, feature_extractor.maxpool
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             feature_extractor.avgpool
                                             )

                feature_channel = self.cnn_encoder[-2][-1].conv1.weight.shape[1]

                # branch of similarity
                self.head_bos = nn.Sequential(
                    nn.Flatten(),
                    # nn.BatchNorm1d(int(feature_channel * bos_c_multiplier)),
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1])
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
                    nn.Flatten(),
                    # nn.BatchNorm1d(int(feature_channel * bom_c_multiplier)),
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))

            elif 'mobilenet' in backbone:
                self.cnn_encoder = nn.Sequential(feature_extractor.features, feature_extractor.avgpool, nn.Flatten())
                feature_channel = feature_extractor.features[-1].out_channels
                # branch of similarity
                self.head_bos = nn.Sequential(
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1]))

                # branch of multi-classification
                self.head_bom = nn.Sequential(
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))

        elif backbone.startswith('cbam'):
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4, feature_extractor.maxpool
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             feature_extractor.avgpool
                                             )
                feature_channel = self.cnn_encoder[-2][-1].conv1.weight.shape[1]

                cbam_attention_bos = CBAM(int(feature_channel * bos_c_multiplier), reduction=r)
                cbam_attention_bom = CBAM(int(feature_channel * bom_c_multiplier), reduction=r)

                # branch of similarity
                self.head_bos = nn.Sequential(
                                              cbam_attention_bos,
                                              nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                                              nn.Flatten(),
                                              nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=0.2),
                                              nn.Linear(bom_fc_out[0], bom_fc_out[1]))

                # branch of multi-classification
                self.head_bom = nn.Sequential(
                                              cbam_attention_bom,
                                              nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                                              nn.Flatten(),
                                              nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=0.2),
                                              nn.Linear(bom_fc_out[0], bom_fc_out[1]))

            elif 'mobilenet' in backbone:
                self.cnn_encoder = feature_extractor.features
                feature_channel = feature_extractor.features[-1].out_channels

                cbam_attention_bos = CBAMlayer(int(feature_channel * bos_c_multiplier), r=r)
                cbam_attention_bom = CBAMlayer(int(feature_channel * bom_c_multiplier), r=r)

                # branch of similarity
                self.head_bos = nn.Sequential(
                    cbam_attention_bos,
                    nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                    nn.Flatten(),
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1])
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
                    cbam_attention_bom,
                    nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                    nn.Flatten(),
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))
        else:
            if 'resnet' in backbone:
                if 'resnetsp' in backbone:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.layer1,
                                                     feature_extractor.layer2, feature_extractor.layer3,
                                                     feature_extractor.layer4, feature_extractor.maxpool
                                                     )
                else:
                    self.cnn_encoder = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu, feature_extractor.maxpool,
                                             feature_extractor.layer1, feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4,
                                             feature_extractor.avgpool
                                             )
                feature_channel = self.cnn_encoder[-2][-1].conv1.weight.shape[1]

                se_attention_bos = SELayer(int(feature_channel * bos_c_multiplier), r=r)
                se_attention_bom = SELayer(int(feature_channel * bom_c_multiplier), r=r)

                # branch of similarity
                self.head_bos = nn.Sequential(
                                              se_attention_bos,
                                              nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                                              nn.Flatten(),
                                              # nn.BatchNorm1d(int(feature_channel*bos_c_multiplier)),
                                              nn.Linear(int(feature_channel* bos_c_multiplier), bos_fc_out[0]),
                                              nn.BatchNorm1d(bos_fc_out[0]),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(bos_fc_out[0], bos_fc_out[1])
                                              # nn.Sigmoid()
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
                                              se_attention_bom,
                                              nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                                              nn.Flatten(),
                                              # nn.BatchNorm1d(int(feature_channel * bom_c_multiplier)),
                                              nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                                              nn.BatchNorm1d(bom_fc_out[0]),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(bom_fc_out[0], bom_fc_out[1]))

            elif 'mobilenet' in backbone:
                self.cnn_encoder = feature_extractor.features
                feature_channel = feature_extractor.features[-1].out_channels

                se_attention_bos = SELayer(int(feature_channel * bos_c_multiplier), r=r)
                se_attention_bom = SELayer(int(feature_channel * bom_c_multiplier), r=r)

                # branch of similarity
                self.head_bos = nn.Sequential(
                    se_attention_bos,
                    nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                    nn.Flatten(),
                    nn.Linear(int(feature_channel * bos_c_multiplier), bos_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bos_fc_out[0], bos_fc_out[1])
                    # nn.Sigmoid()
                )
                # branch of multi-classification
                self.head_bom = nn.Sequential(
                    se_attention_bom,
                    nn.AdaptiveAvgPool2d((1, 1)),  # apply global average pooling to reduce spatial resolution to (1,1)
                    nn.Flatten(),
                    nn.Linear(int(feature_channel * bom_c_multiplier), bom_fc_out[0]),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(bom_fc_out[0], bom_fc_out[1]))

    def forward(self, x_1, x_2):
        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        if self.output_form == 'dual':
            feature_2 = self.cnn_encoder(x_2)
            # print(feature_2.shape)
            if x_1 is not None:
                feature_1 = self.cnn_encoder(x_1)
                # predict with branch of similarity: binary output (dimension=2)
                feature_concat = torch.cat([feature_1, feature_2], dim=1)
                logits_output_bos = self.head_bos(feature_concat)
            else:
                logits_output_bos = None
            # predict with branch of multi-classification: K-class output (dimension=K)
            logits_output_bom = self.head_bom(feature_2)
        elif self.output_form == 'dual2':
            feature_2 = self.cnn_encoder(x_2)
            feature_1 = self.cnn_encoder(x_1)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)

            # predict with branch of similarity: binary output (dimension=2)
            logits_output_bos = self.head_bos(feature_concat)
            # predict with branch of multi-classification: K-class output (dimension=K)
            logits_output_bom = self.head_bom(feature_concat)

        elif self.output_form == 'mclass':
            feature_2 = self.cnn_encoder(x_2)
            # print(feature_2.shape)
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = None

            # predict with branch of multi-classification: K-class output (dimension=K)
            logits_output_bom = self.head_bom(feature_concat)

        else:
            print('Not implemented')
            assert False

        return logits_output_bos, logits_output_bom

    def shap_bos(self, x_1, x_2):
        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        feature_2 = self.cnn_encoder(x_2)
        # print(feature_2.shape)
        if x_1 is not None:
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = self.head_bos(feature_concat)
        else:
            logits_output_bos = None
        # predict with branch of multi-classification: K-class output (dimension=K)
        logits_output_bom = self.head_bom(feature_2)

        return logits_output_bos


    def shap_bom(self, x_1, x_2):
        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        feature_2 = self.cnn_encoder(x_2)
        # print(feature_2.shape)
        if x_1 is not None:
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = self.head_bos(feature_concat)
        else:
            logits_output_bos = None
        # predict with branch of multi-classification: K-class output (dimension=K)
        logits_output_bom = self.head_bom(feature_2)

        return logits_output_bom

class MPB3netExplain(MPB3net):
    def forward(self, x_1, x_2):
        # x_1 is always ok sample
        # x_2 can be ok or ng sample

        # encoder generate feature embeddings of dimension (H,W,M)
        feature_2 = self.cnn_encoder(x_2)
        # print(feature_2.shape)
        if x_1 is not None:
            feature_1 = self.cnn_encoder(x_1)
            # predict with branch of similarity: binary output (dimension=2)
            feature_concat = torch.cat([feature_1, feature_2], dim=1)
            logits_output_bos = self.head_bos(feature_concat)
        else:
            logits_output_bos = None
        # predict with branch of multi-classification: K-class output (dimension=K)
        logits_output_bom = self.head_bom(feature_2)

        return torch.cat((logits_output_bos, logits_output_bom), 1)

# if __name__ == '__main__':
#     from ofa.model_zoo import ofa_net
#     from ofa.utils import download_url

#     ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)

    # import os
    # from dataloader.image_resampler import mp_weighted_resampler,ImageLoader
    # from torchvision import transforms
    # from utils.metrics import FocalLoss
    # import pandas as pd
    #
    # rs_img_size = 224
    # img_folder = '/home/robinru/shiyuan_projects/data/aoi_defect_data_20220906'
    # annotation_filename = os.path.join(img_folder, f'annotation_labels.csv')
    # alltype_annotation_df = pd.read_csv(annotation_filename, index_col=0)
    #
    # Xpairs_resampled, ypair_resampled, position_resampled = mp_weighted_resampler(alltype_annotation_df)
    # # convert to pytorch dataset
    # smtdataset = ImageLoader(img_folder, Xpairs_resampled, ypair_resampled, position_resampled,
    #                          transform=transforms.Compose([
    #                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                   std=[0.229, 0.224, 0.225]),
    #                              transforms.Resize((rs_img_size, rs_img_size)),]))
    #
    # data_loader = torch.utils.data.DataLoader(smtdataset, batch_size=8,
    #                                           shuffle=True, num_workers=1, pin_memory=True)
    # X1, X2, y1, y2, position = next(iter(data_loader))
    #
    # # 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # # 'resnet152
    # model = MPB3net(backbone='resnet101').cuda()
    # X1 = X1.cuda()
    # X2 = X2.cuda()
    # y1 = y1.cuda()
    # y2 = y2.cuda()
    # output_bos, output_bom = model(None, X2)
    # label_binary = (y1==y2).type(torch.int64)
    # criterion = FocalLoss(gamma=0.2, size_average=True)
    # # loss = (criterion(output_bos, label_binary) + criterion(output_bom, y2)) / 2