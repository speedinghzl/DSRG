#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
import findcaffe
import caffe
import argparse
import pprint
import numpy as np
import sys
import os
sys.setrecursionlimit(1500)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--snapshot', dest='snapshot_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, pretrained_model=None, snapshot_model=None):
        """Initialize the SolverWrapper."""

        self.solver = caffe.SGDSolver(solver_prototxt)
        if snapshot_model is not None:
            self.solver.restore(snapshot_model)
        elif pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

    def train_model(self):
        """Network training loop."""
        self.solver.solve()


class Arg(object):
    pass


if __name__ == '__main__':
    args = parse_args()

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    sw = SolverWrapper(args.solver,
                       pretrained_model=args.pretrained_model, snapshot_model=args.snapshot_model)

    print 'Solving...'
    sw.train_model()
    print 'done solving'
