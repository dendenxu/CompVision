from eigenface import *


def train():
    path = "./BioFaceDatabase/BioID-FaceDatabase-V1.2"
    imgext = ".pgm"
    txtext = ".eye"
    if len(sys.argv) > 1:
        path = sys.argv[1]  # the first arg should be the path
    if len(sys.argv) > 2:
        imgext = sys.argv[2]  # the second arg should be image extension
    if len(sys.argv) > 3:
        txtext = sys.argv[3]  # the third should be the eyes position's text file's ext
    mask = EigenFace()
    mask.loadConfig("builtin.json")
    mask.train(path, imgext, txtext, "model.npz", True)


if __name__ == "__main__":
    train()
