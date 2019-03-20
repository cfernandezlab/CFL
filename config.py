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

# Where the weights are located
WEIGHTS_DIR = check(os.path.join(CFL_DIR, 'Weights/'))

# Where the results are saved
RESULTS_DIR = check(os.path.join(CFL_DIR, 'Results/'))

# Folder with datasets
DATASETS_DIR = check(os.path.join(CFL_DIR, 'Datasets/'))

# ---------------------------------------------------------------

## Configuration of CFL 

# Mean color to subtract before propagating an image through a DNN
MEAN_COLOR = [103.939, 116.779, 123.68] 

parser = argparse.ArgumentParser()

# The dataset you want to train/test the model on
parser.add_argument('--dataset', default='SUN360', choices=['SUN360','rot','trans','STANFORD'])

# CFL architecture
parser.add_argument('--network', default='StdConvs', choices=['StdConvs','EquiConvs'])

# GPU to be used
parser.add_argument('--gpu', default="0")

# Ignore missing params 
parser.add_argument('--ignore', action="store_true", default=True)

# TEST config
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--im_height", default=128, type=int)
parser.add_argument("--im_width", default=256, type=int)
parser.add_argument("--im_ch", default=3, type=int)

parser.add_argument("--weight_decay", default=0.0005, type=int)

args = parser.parse_args()

