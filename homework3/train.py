#! python
# python train.py smallSet .pgm .eye builtin.json
# python train.py "Caltec Database -faces" .jpg .txt builtin.json model.color.npz
# python train.py "BioFace Database/BioID-FaceDatabase-V1.2" .pgm .eye builtin.json model.grayscale.npz
from eigenface import *
from utils import *
import argparse
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def help():
    log.info("""
Usage:
python test.py [<dataPath> [<imageExt> [<eyesExt> [<configFileName> [modelName]]]]]

We're assuming a <imageFileNameNoExt>.txt for eye position like
474 247 607 245
Comment line starting with # will be omitted
Or we'll use OpenCV's haarcascade detector to try locating eyes' positions
""")
    log.info("""
Default parameters are:
    path = "./BioFace Database/BioID-FaceDatabase-V1.2"
    imgext = ".pgm"
    txtext = ".eye"
    config = "./default.json"
    modelName = "model.npz"
note that modelName should end with .npz
else the final model would be <modelName>.npz
""")


def train(path, imgext, txtext, config, modelName):
    mask = EigenFace()
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
    parser.add_argument("-p", "--path", default="Caltec Database -faces", help="The image we want to reconstruct and recognize on")
    parser.add_argument("-i", "--imgext", default=".jpg", help="The image we want to reconstruct and recognize on")
    parser.add_argument("-t", "--txtext", default=".txt", help="The image we want to reconstruct and recognize on")
    parser.add_argument("-m", "--model", default="model.color.npz", help="The model trained with this eigenface utility")
    parser.add_argument("-c", "--config", default="builtin.json", help="The configuration file for the eigenface utility instance")

    args = parser.parse_args()
    train()
