import argparse
import os

## PATHS: Please, change this before execution if needed.

# Creates a directory in case it doesn't exist
def check(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

# The project directory
CFL_DIR = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------

## Configuration of CFL 

# Mean color to subtract before propagating an image through a DNN
MEAN_COLOR = [103.939, 116.779, 123.68] 

parser = argparse.ArgumentParser()

# The dataset you want to train/test the model on
parser.add_argument('--dataset', required=True, type=str, help='Path to dataset folders. It must contain RGB/, CM_gt/ and EM_gt/.')

# CFL architecture
parser.add_argument('--network', default='StdConvs', choices=['StdConvs','EquiConvs'], help='CFL architecture')

# Path to weights 
parser.add_argument('--weights', required=True, help= 'Path to weights (eg. weights/StdConvs.ckpt')

# Path to results folder 
parser.add_argument('--results', default=os.path.join(CFL_DIR, 'results/'), help= 'Path to results folder. It will generate the folder if it does not exist.')

# GPU to be used
parser.add_argument('--gpu', default="0", help= 'GPU to be used')

# Ignore missing params 
parser.add_argument('--ignore', action="store_true", default=False, help= 'Ignore missing params')

# TEST config
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--im_height", default=128, type=int)
parser.add_argument("--im_width", default=256, type=int)
parser.add_argument("--im_ch", default=3, type=int)

# TRAIN config
parser.add_argument("--weight_decay", default=0.0005, type=int)

args = parser.parse_args()

