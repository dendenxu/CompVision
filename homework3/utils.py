#! python
from eigenface import *


def faces(config, modelName, mask=None):

    if mask is None:
        # instantiate new eigenface class
        mask = EigenFace()
        mask.loadConfig(config)
        # load previous eigenvectors/mean value
        mask.loadModel(modelName)

    mask.getEigenFaces()

    mean = mask.getMeanFace()
    cv2.imwrite("eigenmean.png", mean)

    # ! Showing only first 12 faces if more is provided
    rows = 3
    cols = 4
    faceCount = rows * cols
    canvas = mask.getCanvas(rows, cols)
    cv2.imwrite("eigenfaces.png", canvas)

    # # ! dangerous, the file might get extremely large
    # allfaces = mask.getCanvas()
    # cv2.imwrite("alleigenfaces.png", allfaces)

    # !should we?
    if not mask.useHighgui:
        # plt.figure(figsize=(10, 10))
        if mask.isColor:
            plt.imshow(mean[:, :, ::-1])  # double imshow
        else:
            plt.imshow(mean, cmap="gray")
        # plt.savefig("eigenmeanfigure.png")
        plt.show()
        if mask.isColor:
            plt.imshow(canvas)
        else:
            plt.imshow(canvas, cmap="gray")
        plt.show()
    else:
        window = f"Mean EigenFaces of {mask.eigenFaces.shape[0]}"
        cv2.imshow(window, mean)
        cv2.waitKey()
        cv2.destroyWindow(window)
        window = f"First {faceCount} EigenFaces"
        cv2.imshow(window, canvas)
        cv2.waitKey()
        cv2.destroyWindow(window)


if __name__ == "__main__":
    # command line arguments
    modelName = "./model.npz"
    config = "./default.json"
    if len(sys.argv) > 1:
        modelName = sys.argv[1]
    if len(sys.argv) > 2:
        config = sys.argv[2]
    faces(config, modelName)
