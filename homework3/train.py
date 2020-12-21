#! python
# python train.py smallSet .pgm .eye builtin.json
from math import sqrt
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

    mean = np.mean(faces, 0)
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
        window = f"Mean EigenFaces of {faces.shape[0]}"
        cv2.imshow(window, mean)
        cv2.waitKey()
        cv2.destroyWindow(window)

    # plt.imshow(mean[:, :, ::-1])
    # plt.show()
    cv2.imwrite("eigenmean.png", mean)

    # ! Showing only first 12 faces if more is provided
    cols = 4
    rows = 3
    faceCount = min(faces.shape[0], cols*rows)
    canvas = np.zeros((rows * faces.shape[1], cols * faces.shape[2]), dtype="uint8")
    for index in range(faceCount):
        i = index // cols
        j = index % cols
        canvas[i * faces.shape[1]:(i+1)*faces.shape[1], j * faces.shape[2]:(j+1)*faces.shape[2]] = mask.normalizeFace(faces[index])
        log.info(f"Filling EigenFace of {index} at {i}, {j}")

    if not mask.useHighgui:
        if mask.isColor:
            plt.imshow(canvas)
        else:
            plt.imshow(canvas, cmap="gray")
        plt.show()
    else:
        window = f"First {faceCount} EigenFaces"
        cv2.imshow(window, canvas)
        cv2.waitKey()
        cv2.destroyWindow(window)

    cv2.imwrite("eigenfaces.png", canvas)


if __name__ == "__main__":
    help()
    train()
