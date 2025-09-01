import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from .rescbam import Res_CBAM_block
from abc import ABC
import torch_dct as DCT
from einops import rearrange
from torch import einsum
# import Activation generator
from .activation_generator import InfMAEFeatureExtractor, InfMAEFeatureProcessor
# import Energy accumulation processor
from .energy_accumulation import EnergyProcessor
# import Peak point generator
from .peak_point_generator import PeakPointGenerator
# import SAM processor
from .sam_processor import SAMProcessor

# Backbone
class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        
        ###Commented out
        # self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # self.C3_n3 = CSPLayer(
        #     int(2 * in_channels[0] * width),
        #     int(in_channels[1] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise = depthwise,
        #     act = act,
        # )
        # self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # self.C3_n4 = CSPLayer(
        #     int(2 * in_channels[1] * width),
        #     int(in_channels[2] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise = depthwise,
        #     act = act,
        # )
       
    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        
        # P3_downsample   = self.bu_conv2(P3_out) 
        # P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        # P4_out          = self.C3_n3(P3_downsample) 
        # P4_downsample   = self.bu_conv1(P4_out)
        # P4_downsample   = torch.cat([P4_downsample, P5], 1)
        # P5_out          = self.C3_n4(P4_downsample)
        # print(P3_out.shape,P4_out.shape,P5_out.shape) #orch.Size([4, 128, 64, 64]) torch.Size([4, 256, 32, 32]) torch.Size([4, 512, 16, 16])
        # P4_out = P4_out.view(P4_out.size(0), -1, P3_out.size(2), P3_out.size(3))
        # P5_out = P5_out.view(P5_out.size(0), -1, P3_out.size(2), P3_out.size(3))
        # return (P3_out, P4_out, P5_out)
        return P3_out

# Detection Head
class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   Input description
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   Use 1x1 convolution for channel integration
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   Use two convolution-normalization-activation for feature extraction
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   Determine the class of feature points
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   Use two convolution-normalization-activation for feature extraction
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   Regression coefficients of feature points
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   Determine if feature points correspond to objects
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

# Neck for long-short term motion-aware learning
class Neck(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        # Key frame and reference frame fusion
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1, act='sigmoid')
        )
        self.conv_cur = BaseConv(channels[0], channels[0],3,1)
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.resblock0 = Res_CBAM_block(in_channels=channels[0]*2,out_channels=channels[0])
        self.freqenhance = FreqEnhance()
        self.temporal_injector = Temporal_injector()
        self.mil_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Output 1 dimension (foreground score)
            nn.Sigmoid()  # Normalize to [0,1] range
        )
        

    def forward(self, feats, targets=None, obj_nums=None, type='val'):
        f_feats = []   # 5* 4,128,64,64
        # Long-short Term Motion-aware Learning
        # Long-term motion
        rc_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(rc_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        # Short-term motion
        # DCT Transform
        fre3 = dct_grid(feats[2])
        fre4 = dct_grid(feats[3])
        fre5 = dct_grid(feats[4])
        # Frequency-domain Enhancement
        freq3 = self.freqenhance(fre3, fre4)
        freq4 = self.freqenhance(fre4, fre5)
        # Temporal Injection
        cur_feat1 = self.temporal_injector(freq3, freq4)
        cur_feat2 = self.resblock0(torch.cat([c_feat, cur_feat1], dim=1))
        f_feats.append(cur_feat2)

        if type == 'train':
            # targets is a list, some elements may be empty
            # Extract ROI features
            roi_features = extract_roi_features(feats[-1], targets)  # 4 128 64 64
            # Multi-instance Learning based on ROI Features
            scores=[]
            for roi_feat in roi_features:
                tmp_scores = []
                N = roi_feat.shape[0]
                if N == 0:
                    tmp_scores.append(torch.empty((0, 1), dtype=torch.float32, device=roi_feat.device))
                else:
                    for i in range(N):
                        temp_roi = roi_feat[i,:,:,:].unsqueeze(0)
                        temp_score = self.mil_classifier(temp_roi)
                        tmp_scores.append(temp_score)
                tmp_scores = torch.cat(tmp_scores, dim=0)
                scores.append(tmp_scores)
            # Filter Top-K Targets to obtain the final pseudo-labels
            filtered_targets = filter_top_k_targets(targets, scores, obj_nums)
            # Compute MIL loss
            smooth_mil_loss = batch_smooth_topk_mil_loss(scores, obj_nums)
            # ablation
            # contrast_mil_loss = batch_contrastive_positive_only_loss(roi_features, scores, obj_nums)
            # Pseudo-label Contrastive Learning (PCL): Compute Contrastive MIL Loss
            contrast_mil_loss = batch_contrastive_mil_loss(roi_features, scores, obj_nums)
            mil_loss = smooth_mil_loss + contrast_mil_loss

            return f_feats, filtered_targets, mil_loss
        else:
            return f_feats

# Our proposed Network (WeCoL)
class slowfastnet(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5):
        super(slowfastnet, self).__init__()
        self.num_frame = num_frame
        self.backbone = YOLOPAFPN(0.33,0.50)

        #-----------------------------------------#
        #   Scale-aware module
        #-----------------------------------------#
        self.neck = Neck(channels=[128,256,512], num_frame=num_frame)
        #----------------------------------------------------------#
        #   Detection head
        #----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")

        # Activation generator (pretrained InfMAE)
        self.infmae_extractor = InfMAEFeatureExtractor('model_data/InfMAE.pth')
        # InfMAE Feature processor
        self.infmae_processor = InfMAEFeatureProcessor()

        # Energy accumulation unit
        self.energy_accumulation = EnergyProcessor()

        # Peak point generator
        self.peak_generator = PeakPointGenerator(target_size=(512, 512), kernel_size=8)

        # SAM processor to generate initial pseudo-labels
        self.sam_processor = SAMProcessor('nets/segment_anything/pretrained/sam_vit_h_4b8939.pth', model_type='vit_h')


    def forward(self, inputs, targets=None, obj_nums=None, type='train'):  #input=[4,3,5,512,512]  B C N H W
        feat = []
        # Extract the multi-frame features from the backbone
        for i in range(self.num_frame):
            feat.append(self.backbone(inputs[:,:,i,:,:]))

        if self.neck and type == 'train':
            # Use Energy accumulation processor to enhance multi-frame features
            feat_tensor = torch.stack(feat, dim=1)  # (B, frames, C, H, W)
            enhanced_features = self.energy_accumulation(feat_tensor)  # (B, C, H, W)

            # Extract Activation Map from Activation Generator
            key_frame = inputs[:, :, -1, :, :]  # (B, 3, 512, 512)
            infmae_features, _ = self.infmae_extractor.extract_features(key_frame)  # (B, 49, 768), (B, 196)
            # Process InfMAE features to match the shape of backbone features
            processed_infmae_features = self.infmae_processor(infmae_features)  # (B, 128, 64, 64)
            # Add processed activation map with enhanced features from energy accumulation to obtain the final activation map
            combined_features = enhanced_features + processed_infmae_features

            # Generate peak points as SAM prompt
            peak_points = self.peak_generator(combined_features, obj_nums)  # List of (B, N, 3)
            # Use SAM to process image and generate pseudo-labels
            with torch.no_grad():
                targets = self.sam_processor(inputs[:, :, -1, :, :], peak_points)  # List of (B, N, 5)
            # Use Neck for long-short term motion-aware learning(LTM)
            feat, newtargets, milloss = self.neck(feat, targets, obj_nums, type)
            outputs  = self.head(feat)
            return outputs, newtargets, milloss
        elif self.neck and type == 'val':
            # Only use Neck for long-short term motion-aware learning during inference
            feature = self.neck(feat, type='val')
            outputs  = self.head(feature)
            return outputs
# Filter top-k targets to obtain the final pseudo-labels
def filter_top_k_targets(targets, scores, obj_num):
    """
    Filter targets based on scores, keeping only the top K pseudo-labels per batch
    :param targets: list[B], each batch has pseudo-labels with shape (N, 5)
    :param scores: list[B], each batch has foreground probabilities (N, 1)
    :param obj_num: list[B], number of targets K per batch
    :return: filtered_targets, filtered pseudo-labels (list[B])
    """
    filtered_targets = []

    for b in range(len(targets)):
        target_b = targets[b]  # Current batch pseudo-labels (N, 5)
        score_b = scores[b]  # Current batch foreground probabilities (N, 1)
        K = obj_num[b][0]  # Number of targets K

        if len(target_b) == 0 or K == 0:
            filtered_targets.append(torch.empty((0, 5), dtype=torch.float32, device=target_b.device))
            continue

        # Sort by score in descending order
        sorted_indices = torch.argsort(score_b.squeeze(), descending=True)  # (N,)
        if sorted_indices.dim() == 0:
            sorted_indices = sorted_indices.unsqueeze(0)
        top_k_indices = sorted_indices[:K]  # Take top K indices

        # Filter out top K targets with highest scores
        filtered_targets.append(target_b[top_k_indices])

    return filtered_targets
# Smooth Top-K MIL Loss
def batch_smooth_topk_mil_loss(scores_list, obj_num, alpha=1.5):
    """
    Compute batch version of Smooth Top-K MIL Loss
    :param scores_list: list[B], scores for each image in batch, shape=(N, 1)
    :param obj_num: list[B], number of targets K for each image in batch
    :param alpha: Softmax smoothing parameter
    :return: batch_loss (average MIL Loss across all images)
    """
    batch_loss = 0
    val_num = 0
    for b in range(len(scores_list)):  # Iterate through all images in batch
        scores = scores_list[b]  # Current image scores (N, 1)
        K = obj_num[b][0]  # Number of targets K for current image

        if scores.numel() == 0 or K == 0:  # Handle case with no pseudo-labels
            continue

        weights = F.softmax(scores.squeeze() * alpha, dim=0).unsqueeze(0)  # Compute Softmax weights
        top_k_weights, _ = torch.topk(weights, min(K, len(weights)))  # Take top K

        loss = -torch.mean(torch.log(top_k_weights + 1e-6))  # Compute Smooth Top-K MIL Loss
        batch_loss += loss
        val_num += 1
    return batch_loss / max(val_num, 1)  # Compute average loss within batch
# Contrastive Loss (Pseudo-label Contrastive Learning)
def batch_contrastive_mil_loss(roi_features_list, scores_list, obj_num, tau=0.07):
    """
    Compute batch-level Contrastive MIL Loss using cosine similarity instead of dot product
    :param roi_features_list: list[B], ROI features for each image in batch (N, C, H, W)
    :param scores_list: list[B], scores for each image in batch (N, 1)
    :param obj_num: list[B], number of targets K for each image in batch
    :param tau: temperature parameter for contrastive learning
    :return: batch_contrastive_loss
    """
    batch_pos_features = []  # Store all Top-K selected ROIs within batch
    batch_neg_features = []  # Store all negative sample ROIs within batch
    valid_count = 0  # Count samples with ROIs in batch

    for b in range(len(scores_list)):
        scores = scores_list[b]  # Current image scores (N, 1)
        roi_features = roi_features_list[b]  # Current image ROI features (N, C, H, W)
        K = obj_num[b][0]  # Number of targets K for current image

        if scores.numel() == 0 or K == 0 or roi_features.numel() == 0:  # Skip invalid images
            continue

        # 1. Select Top-K ROIs for current image
        _, topk_indices = torch.topk(scores.squeeze(), min(K, len(scores)))
        pos_features = roi_features[topk_indices]  # Selected foreground ROIs

        # Ensure pos_features has consistent dimensions
        if pos_features.dim() == 3:  # When becomes (C, H, W), add a dimension
            pos_features = pos_features.unsqueeze(0)

        # 2. Select background ROIs (if exist)
        neg_mask = torch.ones(len(roi_features), dtype=torch.bool, device=roi_features.device)
        neg_mask[topk_indices] = False
        neg_features = roi_features[neg_mask]  # Selected background ROIs

        if pos_features.numel() > 0:  # Store only when pos_features exist
            batch_pos_features.append(pos_features)
        if neg_features.numel() > 0:  # Store only when neg_features exist
            batch_neg_features.append(neg_features)

        valid_count += 1

    # 3. Compute contrastive loss across entire batch
    if valid_count == 0 or len(batch_pos_features) == 0:  # Avoid empty batch
        return torch.tensor(0.0, device=roi_features_list[0].device)

    # Flatten batch_pos_features to (N, D)
    batch_pos_features = torch.cat(batch_pos_features, dim=0)  # (N, C, H, W)
    batch_pos_features = batch_pos_features.view(batch_pos_features.size(0), -1)  # (N, D)

    if len(batch_neg_features) > 0:  # Concatenate only when batch_neg_features exist
        batch_neg_features = torch.cat(batch_neg_features, dim=0)  # (M, C, H, W)
        batch_neg_features = batch_neg_features.view(batch_neg_features.size(0), -1)  # (M, D)
    else:
        batch_neg_features = None  # Set to None when no negative samples

    # Compute cosine similarity (instead of dot product)
    pos_sim = F.cosine_similarity(batch_pos_features.unsqueeze(1), batch_pos_features.unsqueeze(0), dim=-1)  # (N, N)

    if batch_neg_features is not None:
        neg_sim = F.cosine_similarity(batch_pos_features.unsqueeze(1), batch_neg_features.unsqueeze(0), dim=-1)  # (N, M)
        denominator = torch.exp(pos_sim / tau).sum(dim=1) + torch.exp(neg_sim / tau).sum(dim=1) + 1e-6
    else:
        denominator = torch.exp(pos_sim / tau).sum(dim=1) + 1e-6  # Avoid NaN when no negative samples

    # Compute contrastive loss
    contrastive_loss = -torch.log(torch.exp(pos_sim / tau).sum(dim=1) / denominator).mean()

    return contrastive_loss

# Frequency-domain Enhancement for long-short term motion
class FreqEnhance(nn.Module):
    def __init__(self,):
        super(FreqEnhance, self).__init__()

        self.conv_freq1 = two_ConvBnRule(64, 64)
        self.conv_freq2 = two_ConvBnRule(64, 64)
        self.conv_freq3 = two_ConvBnRule(128, 128)

        self.high_band = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)
        self.low_band = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)

    def forward(self, img1_dct, img2_dct):
        # YCbCr
        feat1_y, feat1_Cb, = img1_dct[:, :64, :, :], img1_dct[:, 64:128, :, :] 
        ori_feat1_DCT = img1_dct
        
        feat2_y, feat2_Cb = img2_dct[:, :64, :, :], img2_dct[:, 64:128, :, :]
        
        # high-low freq 
        high1 = torch.cat([feat1_y[:, 32:, :, :], feat1_Cb[:, 32:, :, :]], 1)
        low1 = torch.cat([feat1_y[:, :32, :, :], feat1_Cb[:, :32, :, :]], 1)
        
        high2 = torch.cat([feat2_y[:, 32:, :, :], feat2_Cb[:, 32:, :, :]], 1)
        low2 = torch.cat([feat2_y[:, :32, :, :], feat2_Cb[:, :32, :, :]], 1)
        
        # band-wise freq corr
        high1 = rearrange(high1, 'b c h w -> b c (h w)').transpose(1, 2)
        high2 = rearrange(high2, 'b c h w -> b c (h w)').transpose(1, 2)
        
        low1 = rearrange(low1, 'b c h w -> b c (h w)').transpose(1, 2)
        low2 = rearrange(low2, 'b c h w -> b c (h w)').transpose(1, 2)
        
        high_corr = self.high_band(high2, high1).transpose(1, 2)
        low_corr = self.low_band(low2, low1).transpose(1, 2)
        
        high_corr = rearrange(high_corr, 'b c (h w) -> b c h w', h=img1_dct.shape[-2])
        low_corr = rearrange(low_corr, 'b c (h w) -> b c h w', h=img1_dct.shape[-2])
        
        high_corr = self.conv_freq1(high_corr)
        low_corr = self.conv_freq2(low_corr)
        
        high_y, high_b = torch.split(high_corr, 32, 1)
        low_y, low_b = torch.split(low_corr, 32, 1)
        
        feat_y = torch.cat([low_y, high_y], 1)
        feat_Cb = torch.cat([low_b, high_b], 1)
        
        
        feat_DCT = torch.cat((feat_y, feat_Cb), 1) # concat
        feat_DCT = self.conv_freq3(feat_DCT) + ori_feat1_DCT
        
        return feat_DCT
# Temporal Injector for short-term motion
class Temporal_injector(nn.Module):
    def __init__(self, ):
        super(Temporal_injector, self).__init__()

        self.conv_temp0 = two_ConvBnRule(128, 64)
        self.conv_temp1 = two_ConvBnRule(128, 64)
        self.conv_temp2 = two_ConvBnRule(64, 128)

        self.injector = Transformer(dim=64, heads=2, dim_head=32, mlp_dim=64, dropout=0.)
        
    def forward(self, feat_sam, feat_seq):

        # feat_seq = F.interpolate(feat_seq, size=feat_sam.shape[-2:], mode='bilinear', align_corners=True)
        
        feat_sam = self.conv_temp0(feat_sam)
        feat_seq = self.conv_temp1(feat_seq)
        
        feat_sam = rearrange(feat_sam, 'b c h w -> b c (h w)').transpose(1,2)
        feat_seq = rearrange(feat_seq, 'b c h w -> b c (h w)').transpose(1,2)
        
        feat = self.injector(feat_sam, feat_seq).transpose(1, 2)
        feat = rearrange(feat, 'b c (h w) -> b c h w', h=64)
        feat = self.conv_temp2(feat) 
        
        return feat
# ROI Feature Extraction
def extract_roi_features(image_feature, pseudo_labels):
    """
    Extract ROI features corresponding to pseudo-labels from feature maps, maintaining list format
    :param image_feature: (B, 128, 64, 64) - CNN extracted feature maps
    :param pseudo_labels: list[B], pseudo-labels for each batch with shape (N, x_min, y_min, w, h, c)
    :param img_size: (H_img, W_img) original image size (e.g., 512x512)
    :return: list[B] of roi_features, corresponding to pseudo_labels structure
    """
    B, C, H, W = image_feature.shape  # Feature map size (B, 128, 64, 64)
    H_img, W_img = (512,512)  # Original image size (e.g., 512x512)
    scale_x, scale_y = W / W_img, H / H_img  # Calculate scaling ratio

    roi_features_list = []  # Store batch-level ROI features

    # First filter out pseudo-labels where w,h are greater than 25 to get filtered pseudo-labels



    for b in range(B):
        if len(pseudo_labels[b]) == 0:
            roi_features_list.append(torch.empty((0, C, 7, 7), dtype=torch.float32))  # Keep list length consistent
            continue  # Skip current batch

        # Get current batch pseudo-labels
        pseudo_boxes = pseudo_labels[b] # (N, 5)
        x_min, y_min, w, h, c = pseudo_boxes[:, 0], pseudo_boxes[:, 1], pseudo_boxes[:, 2], pseudo_boxes[:, 3], pseudo_boxes[:, 4]

        # Map to 64x64 feature map proportionally
        x_min, y_min = x_min * scale_x, y_min * scale_y
        x_max, y_max = (x_min + w * scale_x), (y_min + h * scale_y)

        # Form format required by ROI Align (batch_index, x_min, y_min, x_max, y_max)
        batch_index = torch.full_like(x_min, b, dtype=torch.float32)
        rois = torch.stack([batch_index, x_min, y_min, x_max, y_max], dim=1)  # (N, 5)

        # Perform ROI Align
        roi_features = ops.roi_align(image_feature, rois, output_size=(8, 8))  # (N, 128, 7, 7)

        roi_features_list.append(roi_features)  # Store in list

    return roi_features_list  # list[B], each element is (N, 128, 7, 7)
# Transformer for Temporal Injection
class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        
        self.norm = nn.LayerNorm([dim])
        self.cross_attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        
        self.net = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, mlp_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x_q, x_kv):
        
        x = self.cross_attn(x_q, x_kv) 
        x = self.net(x) 
        
        return self.norm(x)
# Attention for Transformer
class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(heads * dim_head, heads * dim_head, bias=False)
        self.to_kv = nn.Linear(heads * dim_head, heads * dim_head * 2, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() 
        
    def forward(self, x_q, x_kv):

        b, c, _, h = *x_q.shape, self.heads
        q = self.to_q(x_q)
        q = rearrange(q, 'b c (h d) -> b h c d', h=h)
        
        kv = self.to_kv(x_kv).chunk(2, dim=-1) 
        k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h=h), kv) 
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale 

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h c d -> b c (h d)')
        out = self.to_out(out)
        
        return out  
# DCT Transform for Frequency-domain Enhancement
def dct_grid(img_ycbcr, grid=8):
    num_batchsize, c, h, w = img_ycbcr.shape
    img_ycbcr = img_ycbcr.reshape(num_batchsize, c, h // grid, grid, w // grid, grid).permute(0, 2, 4, 1, 3, 5)
    img_freq = DCT.dct_2d(img_ycbcr, norm='ortho')
    img_freq = img_freq.reshape(num_batchsize, h // grid, w // grid, -1).permute(0, 3, 1, 2)
    img_freq = img_freq.reshape(num_batchsize, c, h, w) ###add
    return img_freq
# Two ConvBnRule for Temporal Injector
class two_ConvBnRule(nn.Module):
    def __init__(self, in_chan, out_chan=64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        return feat

if __name__ == "__main__":
    net = Temporal_injector() # 4 128 64 64 ->4 512 64 64
    # net = SwinTransformerBlock3D(128) # 4 128 64 64->4 128 64 64
    a = torch.randn(4, 128,64,64)
    b = torch.randn(4, 128,64,64)
    out = net(a,b)  
    print(out.shape)