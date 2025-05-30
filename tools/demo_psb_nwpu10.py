#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'airplane','ship','storage_tank','baseball_diamond','tennis_court','basketball_court','ground_track_field','harbor','bridge','vehicle')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_iter_100000.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.4
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000001.jpg', '000002.jpg', '000004.jpg','000008.jpg','000009.jpg','000010.jpg','000011.jpg','000013.jpg','000015.jpg','000016.jpg','000017.jpg','000020.jpg',
                '000021.jpg', '000024.jpg','000026.jpg','000028.jpg','000030.jpg','000032.jpg','000033.jpg','000034.jpg','000035.jpg','000036.jpg','000037.jpg','000038.jpg','000040.jpg',
                '000041.jpg','000042.jpg','000043.jpg','000045.jpg','000046.jpg','000048.jpg','000051.jpg','000052.jpg','000053.jpg','000054.jpg','000055.jpg','000062.jpg','000066.jpg',
                '000068.jpg','000069.jpg','000073.jpg','000075.jpg','000076.jpg','000078.jpg','000079.jpg','000081.jpg','000083.jpg','000085.jpg','000086.jpg','000088.jpg','000089.jpg',
                '000092.jpg','000094.jpg','000096.jpg','000097.jpg','000098.jpg','000103.jpg','000106.jpg','000107.jpg','000109.jpg','000110.jpg','000112.jpg','000113.jpg','000114.jpg',
                '000117.jpg','000119.jpg','000120.jpg','000121.jpg','000122.jpg','000127.jpg','000128.jpg','000129.jpg','000130.jpg','000131.jpg','000134.jpg','000135.jpg','000136.jpg','000137.jpg','000138.jpg',
                '000142.jpg','000146.jpg','000148.jpg','000149.jpg','000150.jpg','000151.jpg','000152.jpg','000154.jpg','000156.jpg','000159.jpg','000160.jpg','000162.jpg','000164.jpg','000166.jpg','000168.jpg',
                '000169.jpg','000172.jpg','000174.jpg','000176.jpg','000179.jpg','000180.jpg','000181.jpg','000184.jpg','000185.jpg','000186.jpg','000187.jpg','000192.jpg','000193.jpg','000194.jpg','000196.jpg',
                '000197.jpg','000198.jpg','000204.jpg','000207.jpg','000211.jpg','000213.jpg','000216.jpg','000217.jpg','000218.jpg','000219.jpg','000220.jpg','000221.jpg','000223.jpg','000224.jpg','000229.jpg',
                '000230.jpg','000232.jpg','000233.jpg','000234.jpg','000235.jpg','000237.jpg','000238.jpg','000241.jpg','000242.jpg','000244.jpg','000248.jpg','000249.jpg','000250.jpg','000251.jpg','000252.jpg',
                '000253.jpg','000258.jpg','000259.jpg','000260.jpg','000261.jpg','000263.jpg','000266.jpg','000267.jpg','000268.jpg','000269.jpg','000270.jpg','000271.jpg','000272.jpg','000273.jpg','000274.jpg',
                '000280.jpg','000281.jpg','000282.jpg','000283.jpg','000284.jpg','000285.jpg','000290.jpg','000291.jpg','000292.jpg','000293.jpg','000295.jpg','000296.jpg','000297.jpg','000298.jpg','000299.jpg',
                '000301.jpg','000302.jpg','000303.jpg','000304.jpg','000305.jpg','000306.jpg','000309.jpg','000310.jpg','000311.jpg','000312.jpg','000313.jpg','000314.jpg','000315.jpg','000319.jpg','000321.jpg',
                '000323.jpg','000326.jpg','000327.jpg','000328.jpg','000329.jpg','000330.jpg','000331.jpg','000332.jpg','000333.jpg','000335.jpg','000336.jpg','000337.jpg','000338.jpg','000341.jpg','000342.jpg',
                '000343.jpg','000345.jpg','000346.jpg','000347.jpg','000348.jpg','000350.jpg','000352.jpg','000359.jpg','000360.jpg','000363.jpg','000364.jpg','000365.jpg','000366.jpg','000367.jpg','000368.jpg',
                '000373.jpg','000374.jpg','000376.jpg','000377.jpg','000378.jpg','000380.jpg','000382.jpg','000383.jpg','000384.jpg','000385.jpg','000387.jpg','000391.jpg','000392.jpg','000394.jpg','000398.jpg',
                '000399.jpg','000402.jpg','000403.jpg','000405.jpg','000406.jpg','000409.jpg','000410.jpg','000411.jpg','000412.jpg','000415.jpg','000416.jpg','000417.jpg','000418.jpg','000420.jpg','000421.jpg',
                '000422.jpg','000425.jpg','000426.jpg','000427.jpg','000429.jpg','000430.jpg','000432.jpg','000434.jpg','000435.jpg','000436.jpg','000437.jpg','000438.jpg','000439.jpg','000441.jpg','000442.jpg',
                '000443.jpg','000446.jpg','000447.jpg','000449.jpg','000452.jpg','000453.jpg','000455.jpg','000456.jpg','000458.jpg','000459.jpg','000461.jpg','000462.jpg','000464.jpg','000466.jpg','000467.jpg',
                '000468.jpg','000469.jpg','000470.jpg','000472.jpg','000474.jpg','000478.jpg','000479.jpg','000480.jpg','000484.jpg','000486.jpg','000488.jpg','000489.jpg','000491.jpg','000494.jpg','000497.jpg',
                '000498.jpg','000500.jpg','000504.jpg','000505.jpg','000506.jpg','000508.jpg','000509.jpg','000510.jpg','000511.jpg','000513.jpg','000516.jpg','000517.jpg','000522.jpg','000523.jpg','000524.jpg',
                '000525.jpg','000526.jpg','000527.jpg','000528.jpg','000529.jpg','000530.jpg','000532.jpg','000533.jpg','000535.jpg','000537.jpg','000538.jpg','000542.jpg','000544.jpg','000545.jpg','000546.jpg',
                '000548.jpg','000549.jpg','000550.jpg','000551.jpg','000552.jpg','000554.jpg','000556.jpg','000557.jpg','000558.jpg','000559.jpg','000560.jpg','000562.jpg','000563.jpg','000564.jpg','000565.jpg',
                '000568.jpg','000570.jpg','000572.jpg','000574.jpg','000575.jpg','000576.jpg','000577.jpg','000578.jpg','000581.jpg','000582.jpg','000583.jpg','000584.jpg','000587.jpg','000589.jpg','000591.jpg',
                '000592.jpg','000593.jpg','000594.jpg','000595.jpg','000596.jpg','000598.jpg','000600.jpg','000603.jpg','000605.jpg','000606.jpg','000608.jpg','000609.jpg','000610.jpg','000611.jpg','000612.jpg',
                '000613.jpg','000615.jpg','000616.jpg','000618.jpg','000622.jpg','000624.jpg','000626.jpg','000627.jpg','000628.jpg','000632.jpg','000633.jpg','000636.jpg','000638.jpg','000639.jpg','000641.jpg',
                '000642.jpg','000643.jpg','000644.jpg','000645.jpg','000646.jpg','000648.jpg','000649.jpg','000650.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
        plt.savefig("/home/xiaobing/Documents/py-R-FCN/data/testfigs_psb_0.4/" + im_name)

    plt.show()
