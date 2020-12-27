#! python

# Example:
# python train.py -p "Caltec Database -faces" -i .jpg -t .txt -c builtin.json -m model.color.npz

from eigenface import *
from utils import *
import argparse
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def train(path, imgext, txtext, config, modelName):
    mask = EigenFaceUtils()
    mask.loadConfig(config)
    mask.train(path, imgext, txtext, modelName)
    faces(config, modelName, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
We're assuming a <imageFileNameNoExt>.txt for eye position like
474 247 607 245
Comment line starting with # will be omitted
Or we'll use OpenCV's haarcascade detector to try locating eyes' positions
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--path", default="Caltec Database -faces", help="The path of the data, we're assuming that all files are under this directory")
    parser.add_argument("-i", "--imgext", default=".jpg", help="The extension of the image file like .jpg, .png or even .pgm")
    parser.add_argument("-t", "--txtext", default=".txt", help="The text file extension we want to read eye positions off from, usually .txt. But others will work too")
    parser.add_argument("-c", "--config", default="builtin.json", help="The configuration file for the eigenface utility instance")
    parser.add_argument("-m", "--model", default="model.color.npz", help="The model trained with this eigenface utility")

    args = parser.parse_args()
    train(args.path, args.imgext, args.txtext, args.config, args.model)
