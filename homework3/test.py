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
    dst, similar = mask.reconstruct(img)
    dst = mask.normalizeFace(dst)
    similar = mask.normalizeFace(similar)

    if not mask.useHighgui:
        # plt.figure(figsize=(10, 10))
        plt.subplot(131)
        if mask.isColor:
            plt.imshow(img[:, :, ::-1])
        else:
            plt.imshow(img, cmap="gray")
        plt.subplot(132)
        if mask.isColor:
            plt.imshow(dst)
        else:
            plt.imshow(dst, cmap="gray")
        if mask.isColor:
            plt.imshow(similar)
        else:
            plt.imshow(similar, cmap="gray")
        # plt.savefig("testfigure.png")
        plt.show()
    else:
        if mask.isColor:
            canvas = np.zeros((img.shape[0], img.shape[1]+dst.shape[1]+similar.shape[1], 3), dtype="uint8")
            canvas[:, 0:img.shape[1], :] = img
            canvas[:, img.shape[1]:img.shape[1]+dst.shape[1], :] = dst
            canvas[:, img.shape[1]+dst.shape[1]::, :] = similar
        else:
            canvas = np.zeros((img.shape[0], img.shape[1]+dst.shape[1]+similar.shape[1]), dtype="uint8")
            canvas[:, 0:img.shape[1]] = img
            canvas[:, img.shape[1]:img.shape[1]+dst.shape[1]] = dst
            canvas[:, img.shape[1]+dst.shape[1]::] = similar

        window = "Original | EigenReconstruct | Most Similar"
        cv2.namedWindow(window)
        cv2.imshow(window, canvas)
        cv2.waitKey()
        cv2.destroyWindow(window)

    cv2.imwrite("testresult.png", dst)


if __name__ == "__main__":
    help()
    test()
