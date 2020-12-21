#! python
# python train.py smallSet .pgm .eye builtin.json
from eigenface import *


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
    mask.train(path, imgext, txtext, "model.npz")

    faces = mask.getEigenFaces()

    faceCount = 10
    if faces.shape[0] >= 10:
        mean = np.mean(faces[0:10], 0)  # first 10 eigenfaces' average
    else:
        log.warning(f"We've only got {faces.shape[0]} eigenfaces to get mean value on")
        mean = np.mean(faces, 0)
        faceCount = faces.shape[0]

    mean = np.squeeze(mean)
    mean = mask.normalizeFace(mean)
    log.info(f"Getting mean eigenface\n{mean}\nof shape: {mean.shape}")

    if not mask.useHighgui:
        # plt.figure(figsize=(10, 10))
        if mask.isColor:
            plt.imshow(mean[:, :, ::-1])  # double imshow
        else:
            plt.imshow(mean, cmap="gray")
        # plt.savefig("eigenmeanfigure.png")
        plt.show()
    else:
        window = f"Mean EigenFaces of First {faceCount}"
        cv2.imshow(window, mean)
        cv2.waitKey()
        cv2.destroyWindow(window)

    # plt.imshow(mean[:, :, ::-1])
    # plt.show()
    cv2.imwrite("eigenmean.png", mean)


if __name__ == "__main__":
    help()
    train()
