#! python
from eigenface import *


def train():
    path = "./BioFace Database/BioID-FaceDatabase-V1.2"
    imgext = ".pgm"
    txtext = ".eye"
    config = "builtin.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]  # the first arg should be the path
    if len(sys.argv) > 2:
        imgext = sys.argv[2]  # the second arg should be image extension
    if len(sys.argv) > 3:
        txtext = sys.argv[3]  # the third should be the eyes position's text file's ext
    if len(sys.argv) > 4:
        config = sys.argv[4]
    mask = EigenFace()
    mask.loadConfig(config)
    mask.train(path, imgext, txtext, "model.npz", True)


if __name__ == "__main__":
    train()
