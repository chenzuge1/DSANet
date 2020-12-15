# import torch
# import math
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ..builder import HEADS
from .anchor_head import AnchorHead

# class Self_Attn(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim):
#         super(Self_Attn, self).__init__()
#         self.chanel_in = in_dim
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).contiguous().view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#
#         #新增scale,其中dk是V的向量长度，当其过大时会使得到的点积值过大，从而导致softmax的梯度很小，因此除上dk来改善这种情况。
#         d_k = C
#         energy = torch.bmm(proj_query, proj_key) / math.sqrt(d_k) # transpose check
#         # energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x).contiguous().view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x
#         # return out, attention
#         return out

@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
        # for i in range(2):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)

        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors, 3, padding=1)

        # self.cls_selfatt = Self_Attn(self.feat_channels)
        # self.reg_selfatt = Self_Attn(self.feat_channels)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
        normal_init(self.retina_iou, std=0.01)


        # normal_init(self.cls_selfatt.query_conv, std=0.01)
        # normal_init(self.cls_selfatt.key_conv, std=0.01)
        # normal_init(self.cls_selfatt.value_conv, std=0.01)
        #
        # normal_init(self.reg_selfatt.query_conv, std=0.01)
        # normal_init(self.reg_selfatt.key_conv, std=0.01)
        # normal_init(self.reg_selfatt.value_conv, std=0.01)


    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        # cls_feat = x
        # reg_feat = x
        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        # for reg_conv in self.reg_convs:
        #     reg_feat = reg_conv(reg_feat)
        #
        # # reg_feat, cls_feat = x
        # cls_score = self.retina_cls(cls_feat)
        # bbox_pred = self.retina_reg(reg_feat)
        # # bbox_iou  = self.retina_iou(reg_feat)
        # bbox_iou  = self.retina_iou(cls_feat)
        # return cls_score, bbox_pred, bbox_iou

        cls_feat, reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # if x.shape[2] < 76: # 去除conv3层【76,116】
        #     cls_feat = self.cls_selfatt(cls_feat)
        #     reg_feat = self.reg_selfatt(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        # bbox_iou  = self.retina_iou(reg_feat)
        bbox_iou  = self.retina_iou(cls_feat)

        return cls_score, bbox_pred, bbox_iou
        # return cls_score, bbox_pred



# import torch.nn as nn
# from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
#
# from ..builder import HEADS
# from .anchor_head import AnchorHead
#
#
# @HEADS.register_module()
# class RetinaHead(AnchorHead):
#     r"""An anchor-based head used in `RetinaNet
#     <https://arxiv.org/pdf/1708.02002.pdf>`_.
#
#     The head contains two subnetworks. The first classifies anchor boxes and
#     the second regresses deltas for the anchors.
#
#     Example:
#         >>> import torch
#         >>> self = RetinaHead(11, 7)
#         >>> x = torch.rand(1, 7, 32, 32)
#         >>> cls_score, bbox_pred = self.forward_single(x)
#         >>> # Each anchor predicts a score for each class except background
#         >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
#         >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
#         >>> assert cls_per_anchor == (self.num_classes)
#         >>> assert box_per_anchor == 4
#     """
#
#     def __init__(self,
#                  num_classes,
#                  in_channels,
#                  stacked_convs=4,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  anchor_generator=dict(
#                      type='AnchorGenerator',
#                      octave_base_scale=4,
#                      scales_per_octave=3,
#                      ratios=[0.5, 1.0, 2.0],
#                      strides=[8, 16, 32, 64, 128]),
#                  **kwargs):
#         self.stacked_convs = stacked_convs
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         super(RetinaHead, self).__init__(
#             num_classes,
#             in_channels,
#             anchor_generator=anchor_generator,
#             **kwargs)
#
#     def _init_layers(self):
#         """Initialize layers of the head."""
#         self.relu = nn.ReLU(inplace=True)
#         self.cls_convs = nn.ModuleList()
#         self.reg_convs = nn.ModuleList()
#         for i in range(self.stacked_convs):
#         # for i in range(2):
#             chn = self.in_channels if i == 0 else self.feat_channels
#             self.cls_convs.append(
#                 ConvModule(
#                     chn,
#                     self.feat_channels,
#                     3,
#                     stride=1,
#                     padding=1,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=self.norm_cfg))
#             self.reg_convs.append(
#                 ConvModule(
#                     chn,
#                     self.feat_channels,
#                     3,
#                     stride=1,
#                     padding=1,
#                     conv_cfg=self.conv_cfg,
#                     norm_cfg=self.norm_cfg))
#         self.retina_cls = nn.Conv2d(
#             self.feat_channels,
#             self.num_anchors * self.cls_out_channels,
#             3,
#             padding=1)
#         self.retina_reg = nn.Conv2d(
#             self.feat_channels, self.num_anchors * 4, 3, padding=1)
#         self.retina_iou = nn.Conv2d(
#             self.feat_channels, self.num_anchors, 3, padding=1)
#
#     def init_weights(self):
#         """Initialize weights of the head."""
#         for m in self.cls_convs:
#             normal_init(m.conv, std=0.01)
#         for m in self.reg_convs:
#             normal_init(m.conv, std=0.01)
#         bias_cls = bias_init_with_prob(0.01)
#         normal_init(self.retina_cls, std=0.01, bias=bias_cls)
#         normal_init(self.retina_reg, std=0.01)
#         normal_init(self.retina_iou, std=0.01)
#
#     def forward_single(self, x):
#         """Forward feature of a single scale level.
#
#         Args:
#             x (Tensor): Features of a single scale level.
#
#         Returns:
#             tuple:
#                 cls_score (Tensor): Cls scores for a single scale level
#                     the channels number is num_anchors * num_classes.
#                 bbox_pred (Tensor): Box energies / deltas for a single scale
#                     level, the channels number is num_anchors * 4.
#         """
#         # cls_feat = x
#         # reg_feat = x
#         # for cls_conv in self.cls_convs:
#         #     cls_feat = cls_conv(cls_feat)
#         # for reg_conv in self.reg_convs:
#         #     reg_feat = reg_conv(reg_feat)
#         #
#         # # reg_feat, cls_feat = x
#         # cls_score = self.retina_cls(cls_feat)
#         # bbox_pred = self.retina_reg(reg_feat)
#         # # bbox_iou  = self.retina_iou(reg_feat)
#         # bbox_iou  = self.retina_iou(cls_feat)
#         # return cls_score, bbox_pred, bbox_iou
#
#         cls_feat, reg_feat = x
#         for cls_conv in self.cls_convs:
#             cls_feat = cls_conv(cls_feat)
#         for reg_conv in self.reg_convs:
#             reg_feat = reg_conv(reg_feat)
#
#         cls_score = self.retina_cls(cls_feat)
#         bbox_pred = self.retina_reg(reg_feat)
#         # bbox_iou  = self.retina_iou(reg_feat)
#         bbox_iou  = self.retina_iou(cls_feat)
#
#         return cls_score, bbox_pred, bbox_iou
#         # return cls_score, bbox_pred
