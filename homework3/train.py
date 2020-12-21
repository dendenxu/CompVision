#! python
from eigenface import *


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
    if faces.shape[0] >= 10:
        mean = np.mean(faces[0:10], 0)  # first 10 eigenfaces' average
    else:
        log.warning(f"We've only got {faces.shape[0]} eigenfaces to get mean value on")
        mean = np.mean(faces, 0)

    mean = np.squeeze(mean)
    log.info(f"Getting mean eigenface\n{mean}\nof shape: {mean.shape}")

    if not mask.useHighgui:
        if mask.isColor:
            plt.imshow(mean[:, :, ::-1])  # double imshow
        else:
            plt.imshow(mean, cmap="gray")
        plt.show()
        plt.savefig("eigenmean.png")
    else:
        cv2.imshow(np.clip(mean, 0, 255).astype("uint8"))
        cv2.waitKey()


if __name__ == "__main__":
    train()
