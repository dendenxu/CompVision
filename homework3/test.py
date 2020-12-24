#! python
# python test.py image_0318.jpg model.npz builtin.json
from eigenface import *


def help():
    log.info("""
Usage:
python test.py [<imageFileName> [<modelFileName> [<configFileName>]]]

We're assuming a <imageFileNameNoExt>.txt for eye position like
474 247 607 245
Comment line starting with # will be omitted
Or we'll use OpenCV's haarcascade detector to try locating eyes' positions
""")
    log.info("""
Default parameters are
    imgname = "./test.tiff"
    modelName = "./model.npz"
    config = "./default.json"
""")


def test():
    # process arguments
    imgname = "./test.tiff"
    modelName = "./model.npz"
    config = "./default.json"
    if len(sys.argv) > 1:
        imgname = sys.argv[1]
    if len(sys.argv) > 2:
        modelName = sys.argv[2]
    if len(sys.argv) > 3:
        config = sys.argv[3]

    # instantiate new eigenface class
    mask = EigenFace()
    mask.loadConfig(config)
    # load previous eigenvectors/mean value
    mask.loadModel(modelName)
    txtname = f"{os.path.splitext(imgname)[0]}.txt"
    if os.path.isfile(txtname):
        mask.getDictEntry(txtname)

    log.info(f"Loading image: {imgname}")
    img = mask.getImage(imgname)

    dst, eigen, face, ori = mask.reconstruct(img)

    # ori = mask.normalizeFace(ori)

    cv2.imwrite("testresult.png", dst)
    w = mask.w
    h = mask.h
    if mask.isColor:
        canvas = np.zeros((h, 5*w, 3), dtype="uint8")
    else:
        canvas = np.zeros((h, 5*w), dtype="uint8")

    canvas[:, 0*w:1*w] = img
    canvas[:, 1*w:2*w] = dst
    canvas[:, 2*w:3*w] = eigen
    canvas[:, 3*w:4*w] = face
    canvas[:, 4*w:5*w] = ori

    cv2.imwrite("similar.png", canvas)

    if not mask.useHighgui:
        plt.imshow(canvas)
        plt.show()
    else:

        window = "Original | EigenReconstruct | Most Similar"
        cv2.namedWindow(window)
        cv2.imshow(window, canvas)
        cv2.waitKey()
        cv2.destroyWindow(window)


if __name__ == "__main__":
    help()
    test()
