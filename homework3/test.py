from eigenface import *


def test():
    # process arguments
    imgname = "./image_0001.jpg"
    modelName = "./model.npz"
    if len(sys.argv) > 1:
        imgname = sys.argv[1]
    if len(sys.argv) > 2:
        modelName = sys.argv[2]

    # instantiate new eigenface class
    mask = EigenFace()
    # load previous eigenvectors/mean value
    mask.loadModel(modelName)
    img = mask.getImage(imgname)
    dst = mask.reconstruct(img)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.subplot(122)
    if mask.isColor:
        plt.imshow(dst[:, :, ::-1])
    else:
        plt.imshow(dst, cmap="gray")
    plt.show()


if __name__ == "__main__":
    test()
