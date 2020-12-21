#! python
from eigenface import *


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
    img = mask.getImage(imgname)
    dst = mask.reconstruct(img)

    plt.figure()
    plt.subplot(121)
    if mask.isColor:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img, cmap="gray")
    plt.subplot(122)
    if mask.isColor:
        plt.imshow(np.clip(dst[:, :, ::-1], 0, 255).astype("uint8"))
    else:
        plt.imshow(dst, cmap="gray")
    plt.show()
    plt.savefig("figure.png")


if __name__ == "__main__":
    test()
