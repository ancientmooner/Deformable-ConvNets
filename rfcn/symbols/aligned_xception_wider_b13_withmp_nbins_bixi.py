import cPickle
import numpy as np
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *


class aligned_xception_wider_b13_withmp_nbins_bixi(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-3
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_msra_std(self, shape):
        print shape
        fan_in = float(shape[1])
        if len(shape) > 2:
            fan_in *= np.prod(shape[2:])
        print(np.sqrt(2 / fan_in))
        return np.sqrt(2 / fan_in)

    def separable_conv(self, data, n_in_ch, n_out_ch, kernel, pad, name, depth_mult=1, dilate=(1, 1)):
        dw_out = mx.contrib.sym.SeparableConvolution(data=data, num_filter=n_in_ch, pad=pad, kernel=kernel, dilate=dilate,
                                                     no_bias=True, num_group=n_in_ch, name=name + '_depthwise_kernel')

        #  pointwise convolution
        pw_out = mx.sym.Convolution(dw_out, num_filter=n_out_ch, kernel=(1, 1),
                                    no_bias=True, name=name + '_pointwise_kernel')
        return pw_out

    def get_xception_feat(self, data):
        data = mx.sym.Variable('data')
        b1 = mx.sym.Convolution(data, num_filter=32, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                                no_bias=True, name='block1_conv1')
        b1 = mx.sym.BatchNorm(b1, name='block1_conv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b1 = mx.sym.Activation(b1, act_type='relu', name='block1_conv1_act')
        b1 = mx.sym.Convolution(b1, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                no_bias=True, name='block1_conv2')
        b1 = mx.sym.BatchNorm(b1, name='block1_conv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b1 = mx.sym.Activation(b1, act_type='relu', name='block1_conv2_act')

        # block 2
        rs2 = mx.sym.Convolution(b1, num_filter=128, kernel=(1, 1), stride=(2, 2),
                                 no_bias=True, name='convolution2d_1')
        rs2 = mx.sym.BatchNorm(rs2, name='batchnormalization_1', use_global_stats=self.use_global_stats, fix_gamma=False)

        b2 = self.separable_conv(b1, 64, 128, (3, 3), (1, 1), 'block2_sepconv1')
        b2 = mx.sym.BatchNorm(b2, name='block2_sepconv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b2 = mx.sym.Activation(b2, act_type='relu', name='block2_sepconv1_act')
        b2 = self.separable_conv(b2, 128, 128, (3, 3), (1, 1), 'block2_sepconv2')
        b2 = mx.sym.BatchNorm(b2, name='block2_sepconv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)

        # b2 = mx.sym.Pad(b2, mode='edge', pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
        b2 = mx.sym.Pooling(b2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max',
                            pooling_convention='valid', name='block2_pool')
        b2 = b2 + rs2

        # block 3
        rs3 = mx.sym.Convolution(b2, num_filter=256, kernel=(1, 1), stride=(2, 2),
                                 no_bias=True, name='convolution2d_2')
        rs3 = mx.sym.BatchNorm(rs3, name='batchnormalization_2', use_global_stats=self.use_global_stats, fix_gamma=False)

        b3 = mx.sym.Activation(b2, act_type='relu', name='block3_sepconv1_act')
        b3 = self.separable_conv(b3, 128, 256, (3, 3), (1, 1), 'block3_sepconv1')
        b3 = mx.sym.BatchNorm(b3, name='block3_sepconv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b3 = mx.sym.Activation(b3, act_type='relu', name='block3_sepconv2_act')
        b3 = self.separable_conv(b3, 256, 256, (3, 3), (1, 1), 'block3_sepconv2')
        b3 = mx.sym.BatchNorm(b3, name='block3_sepconv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)

        b3 = mx.sym.Pooling(b3, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max',
                            pooling_convention='valid', name='block3_pool')
        b3 = b3 + rs3

        # block 4
        rs4 = mx.sym.Convolution(b3, num_filter=1024, kernel=(1, 1), stride=(2, 2),
                                 no_bias=True, name='convolution2d_3')
        rs4 = mx.sym.BatchNorm(rs4, name='batchnormalization_3', use_global_stats=self.use_global_stats, fix_gamma=False)

        b4 = mx.sym.Activation(b3, act_type='relu', name='block4_sepconv1_act')
        b4 = self.separable_conv(b4, 256, 512, (3, 3), (1, 1), 'block4_sepconv1')
        b4 = mx.sym.BatchNorm(b4, name='block4_sepconv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b4 = mx.sym.Activation(b4, act_type='relu', name='block4_sepconv2_act')
        b4 = self.separable_conv(b4, 512, 1024, (3, 3), (1, 1), 'block4_sepconv2')
        b4 = mx.sym.BatchNorm(b4, name='block4_sepconv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)

        # b4 = mx.sym.Pad(b4, mode='edge', pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
        b4 = mx.sym.Pooling(b4, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max',
                            pooling_convention='valid', name='block4_pool')
        b4 = b4 + rs4

        b = b4
        for i in range(8):
            residual = b
            prefix = 'block' + str(i + 5)

            b = mx.sym.Activation(b, act_type='relu', name=prefix + '_sepconv1_act')
            b = self.separable_conv(b, 1024, 1024, (3, 3), (1, 1), prefix + '_sepconv1')
            b = mx.sym.BatchNorm(b, name=prefix + '_sepconv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
            b = mx.sym.Activation(b, act_type='relu', name=prefix + '_sepconv2_act')
            b = self.separable_conv(b, 1024, 1024, (3, 3), (1, 1), prefix + '_sepconv2')
            b = mx.sym.BatchNorm(b, name=prefix + '_sepconv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
            b = mx.sym.Activation(b, act_type='relu', name=prefix + '_sepconv3_act')
            b = self.separable_conv(b, 1024, 1024, (3, 3), (1, 1), prefix + '_sepconv3')
            b = mx.sym.BatchNorm(b, name=prefix + '_sepconv3_bn', use_global_stats=self.use_global_stats, fix_gamma=False)

            b = b + residual

        rs5 = mx.sym.Convolution(b, num_filter=1024, kernel=(1, 1), stride=(1, 1),
                                 no_bias=True, name='convolution2d_4')
        rs5 = mx.sym.BatchNorm(rs5, name='batchnormalization_4', use_global_stats=self.use_global_stats, fix_gamma=False)

        b13 = mx.sym.Activation(b, act_type='relu', name='block13_sepconv1_act')
        b13 = self.separable_conv(b13, 1024, 1024, (3, 3), (1, 1), 'block13_sepconv1')
        b13 = mx.sym.BatchNorm(b13, name='block13_sepconv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b13 = mx.sym.Activation(b13, act_type='relu', name='block13_sepconv2_act')
        b13 = self.separable_conv(b13, 1024, 1024, (3, 3), (1, 1), 'block13_sepconv2')
        b13 = mx.sym.BatchNorm(b13, name='block13_sepconv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b13 = mx.sym.Pooling(b13, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max',
                             pooling_convention='valid', name='block13_pool')
        b13 = b13 + rs5

        # b13 = mx.sym.Pad(b13, mode='edge', pad_width=(0, 0, 0, 0, 1, 1, 1, 1))

        return b13

    def get_xception_exit_flow(self, b):

        # b14 = self.separable_conv(b13, 1024, 1536, (3, 3), (1, 1), 'block14_sepconv1')
        b14 = self.separable_conv(b, 1024, 2048, (3, 3), (2, 2), 'block14_sepconv1', dilate=(2, 2))
        b14 = mx.sym.BatchNorm(b14, name='block14_sepconv1_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b14 = mx.sym.Activation(b14, act_type='relu', name='block14_sepconv1_act')
        # b14 = self.separable_conv(b14, 1536, 2048, (3, 3), (1, 1), 'block14_sepconv2')
        b14 = self.separable_conv(b14, 2048, 2048, (3, 3), (2, 2), 'block14_sepconv2', dilate=(2, 2))
        b14 = mx.sym.BatchNorm(b14, name='block14_sepconv2_bn', use_global_stats=self.use_global_stats, fix_gamma=False)
        b14 = mx.sym.Activation(b14, act_type='relu', name='block14_sepconv2_act')

        return b14


    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_xception_feat(data)
        conv_exit_flow = self.get_xception_exit_flow(conv_feat)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)
        rpn_loss_scale = float(cfg.network.RPN_LOSS_SCALE)
        if is_train:
            # prepare rpn data

            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob",
                                                grad_scale=rpn_loss_scale)
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=rpn_loss_scale / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)



        # conv_new_1

        nbins = cfg.network.NBINS
        conv_new_1 = mx.sym.Convolution(data=conv_exit_flow, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=float(cfg.network.CONVNEW_LR_MULT))
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=nbins*nbins*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=nbins*nbins*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=nbins, pooled_size=nbins,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=nbins, pooled_size=nbins,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(nbins, nbins))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(nbins, nbins))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        rcnn_loss_scale = float(cfg.network.RCNN_LOSS_SCALE)
        print rpn_loss_scale, rcnn_loss_scale
        if is_train:

            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=rcnn_loss_scale)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=rcnn_loss_scale / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid', grad_scale=rcnn_loss_scale)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=rcnn_loss_scale / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def get_symbol_rpn(self, cfg, is_train=True):
        # config alias for convenient
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_xception_feat(data)
        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)
		
	rpn_loss_scale = float(cfg.network.RPN_LOSS_SCALE)
        if is_train:
            # prepare rpn data

            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob",
                                                grad_scale=rpn_loss_scale)
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=rpn_loss_scale / cfg.TRAIN.RPN_BATCH_SIZE)
            group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss])
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois, score = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois, score = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            group = mx.symbol.Group([rois, score])
					
        self.sym = group
        return group

    def get_symbol_rfcn(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        if is_train:
            data = mx.symbol.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            label = mx.symbol.Variable(name='label')
            bbox_target = mx.symbol.Variable(name='bbox_target')
            bbox_weight = mx.symbol.Variable(name='bbox_weight')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes), name='bbox_target_reshape')
            bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_reg_classes), name='bbox_weight_reshape')
        else:
            data = mx.sym.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

        # shared convolutional layers
        conv_feat = self.get_xception_feat(data)
        conv_exit_flow = self.get_xception_exit_flow(conv_feat)

        nbins = cfg.network.NBINS
        conv_new_1 = mx.sym.Convolution(data=conv_exit_flow, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=float(cfg.network.CONVNEW_LR_MULT))
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=nbins*nbins*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=nbins*nbins*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=nbins, pooled_size=nbins,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=nbins, pooled_size=nbins,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(nbins, nbins))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(nbins, nbins))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        rcnn_loss_scale = float(cfg.network.RCNN_LOSS_SCALE)
        if is_train:

            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=rcnn_loss_scale)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=float(rcnn_loss_scale) / cfg.TRAIN.BATCH_ROIS_OHEM)
                #rcnn_label = labels_ohem
		label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid', grad_scale=rcnn_loss_scale)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=float(rcnn_loss_scale) / cfg.TRAIN.BATCH_ROIS)
                #rcnn_label = label

            # reshape output
            #rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            #group = mx.sym.Group([cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
	    group = mx.sym.Group([cls_prob, bbox_loss, mx.sym.BlockGrad(label)]) if cfg.TRAIN.ENABLE_OHEM else mx.sym.Group([cls_prob, bbox_loss])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([cls_prob, bbox_pred])
			#group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])


    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight_rfcn(self, cfg, arg_params, aux_params):
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])
