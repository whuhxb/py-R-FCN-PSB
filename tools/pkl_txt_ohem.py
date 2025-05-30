# -*- coding: utf-8 -*-
# change format from "pkl" to "txt" for submitting final results for VOC2012 test
# Usage: python ./tools/results_for_server.py --imdb voc_2007_test

import _init_paths
#from fast_rcnn.test import test_net
#from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
import pickle
import cPickle
from datasets.factory import get_imdb
import argparse
import time, os, sys
import scipy.io as scio

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    imdb = get_imdb(args.imdb_name)

    dets_path = '/home/stu_3/Documents/task1/py-R-FCN-mydataset/py-R-FCN/output/rfcn_end2end_ohem/voc_0712_test/resnet101_rfcn_ohem_iter_80000/'

    det_file = os.path.join(dets_path, 'detections.pkl')
    # det_file = os.path.join(dets_path, 'detections.pkl')
    fdets = open(det_file,'r+')
    all_boxes = pickle.load(fdets)
    
    # saving results for server
    path = '/home/stu_3/Documents/task1/py-R-FCN-mydataset/py-R-FCN/output/rfcn_end2end_ohem/voc_0712_test/resnet101_rfcn_ohem_iter_80000/results/'
    results_path = os.path.join(path, args.imdb_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    scio.savemat(os.path.join(results_path, 'detections.mat'), {'all_boxes':all_boxes, 'image_name':imdb.image_index, 'class_name':imdb.classes})   # "mat"

    num_images = len(imdb.image_index)
    for j in xrange(1, imdb.num_classes):
        # opening "txt" file for saveing results
	txt_name = 'comp4_det_test_' + imdb.classes[j] + '.txt'
	txt_path = os.path.join(results_path, txt_name)
        f = open(txt_path, 'w+')
	for i in xrange(num_images):
	    cls_dets = all_boxes[j][i]
	    for k in xrange(cls_dets.shape[0]):
		"""x1 = str(cls_dets[k,0])
		y1 = str(cls_dets[k,1])
		x2 = str(cls_dets[k,2])
		y2 = str(cls_dets[k,3])
		score = str(cls_dets[k,4])"""
		x1 = "%.6f" % cls_dets[k,0]
		y1 = "%.6f" % cls_dets[k,1]
		x2 = "%.6f" % cls_dets[k,2]
		y2 = "%.6f" % cls_dets[k,3]
		score = "%.6f" % cls_dets[k,4]
		content = imdb.image_index[i]+' '+score+' '+x1+' '+y1+' '+x2+' '+y2+'\n'
		f.write(content)
        f.close()



    

