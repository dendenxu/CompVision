#! python
# python train.py smallSet .pgm .eye builtin.json
from eigenface import *
from faces import *


def help():
    log.info("""
Usage:
python test.py [<dataPath> [<imageExt> [<eyesExt> [<configFileName>]]]]

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
""")


def train():
    path = "./BioFace Database/BioID-FaceDatabase-V1.2"
    imgext = ".pgm"
    txtext = ".eye"
    config = "default.json"
    modelName = "model.npz"
    if len(sys.argv) > 1:
        path = sys.argv[1]  # the first arg should be the path
    if len(sys.argv) > 2:
        imgext = sys.argv[2]  # the second arg should be image extension
    if len(sys.argv) > 3:
        txtext = sys.argv[3]  # the third should be the eyes position's text file's ext
    if len(sys.argv) > 4:
        config = sys.argv[4]
    if len(sys.argv) > 5:
        modelName = sys.argv[5]
    mask = EigenFace()
    mask.loadConfig(config)
    mask.train(path, imgext, txtext, modelName)
    faces(config, modelName, mask)


if __name__ == "__main__":
    help()
    train()
