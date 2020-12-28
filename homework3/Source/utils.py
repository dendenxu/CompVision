#! python
from eigenface import *


def faces(config, modelName, mask=None):

    if mask is None:
        # instantiate new eigenface class
        mask = EigenFaceUtils()
        mask.loadConfig(config)
        # load previous eigenvectors/mean value
        mask.loadModel(modelName)

    mask.updateEigenFaces()

    mean = mask.getMeanEigen()
    cv2.imwrite("eigenmean.png", mean)

    # ! Showing only first 12 faces if more is provided
    rows = 3
    cols = 4
    faceCount = rows * cols
    canvas = mask.drawEigenFaces(rows, cols)
    cv2.imwrite("eigenfaces.png", canvas)

    # # ! dangerous, the file might get extremely large
    # allfaces = mask.getCanvas()
    # cv2.imwrite("alleigenfaces.png", allfaces)

    # !should we?
    if not mask.useHighgui:
        log.error("Only HighGUI of OpenCV is supported.\nOther implementation removed due to regulation.")
    else:
        window = f"Mean EigenFaces of {mask.eigenFaces.shape[0]}"
        cv2.imshow(window, mean)
        cv2.waitKey()
        cv2.destroyWindow(window)
        window = f"First {faceCount} EigenFaces"
        cv2.imshow(window, canvas)
        cv2.waitKey()
        cv2.destroyWindow(window)
