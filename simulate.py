import h5py
from os.path import realpath, dirname, join, isdir
from os import makedirs
import numpy as np
from time import time
from argparse import ArgumentParser
from PIL import Image
import cv2

import sys
script_dir = realpath(dirname(__file__))
bin_dir = join(script_dir, 'bin')
if isdir(bin_dir):
    sys.path.append(bin_dir)

from simulator import DVSSimulator

def parse_args():
    parser = ArgumentParser(description='Construct events from two consecutive images')
    sample_dir = join(script_dir, 'data')
    sample1_path = join(sample_dir, 'frame0000.jpg')
    sample2_path = join(sample_dir, 'frame0001.jpg')
    parser.add_argument('-i1', '--input1',
            help='name of a file with the first image in the stream. \
                    Sample frame is located at {}'.format(sample1_path),
            required=True)
    parser.add_argument('-i2', '--input2',
            help='name of a file with the second image in the stream \
                    Sample frame is located at {}'.format(sample2_path),
            required=True)
    parser.add_argument('-o', '--output',
            help='name of the file to the write events',
            required=True)
    parser.add_argument('-C', help='sensitivity of the sensor',
            type=float, required=False, default=0.15)
    parser.add_argument('-t', '--time',
            help='timestamp of the first image (microseconds)',
            type=int, required=False, default=0)
    parser.add_argument('--fps', help='number of frames per second',
            type=float, required=False, default=25)

    return parser.parse_args(sys.argv[1:])

def sure_dir_exists(dir_name):
    if not isdir(dir_name):
        makedirs(dir_name)
    return dir_name

def load_image(filename):
    #return np.asarray(Image.open(filename).convert('L'))
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def serialize(filename, pol, timestamps, x_pos, y_pos):
    f = h5py.File(filename, 'w')
    f.create_dataset('pol', data=np.array(pol))
    f.create_dataset('timestamps', data=np.array(timestamps))
    f.create_dataset('x_pos', data=np.array(x_pos))
    f.create_dataset('y_pos', data=np.array(y_pos))
    f.close()

def main():
    args = parse_args()

    init_time = int(args.time) # timestamp of the first image
    dt = int(1e6 / args.fps) # time interval between images
    C = args.C
    init_image = load_image(args.input1)
    sim = DVSSimulator(init_image, init_time, C)

    img = load_image(args.input2)
    events = sim.update(img, init_time + dt)

    sure_dir_exists(realpath(dirname(args.output)))
    serialize(args.output, events['polarities'],
            events['timestamps'], events['x_positions'],
            events['y_positions'])

if __name__=='__main__':
    main()
