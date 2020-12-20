from eigenface import *


def test():
    name = "./image_0001.jpg"
    if len(sys.argv) > 2:
        name = sys.argv[1]
    mask = EigenFace()
    data = np.load("model.npz")
    mask.eigenVectors = data["arr_0"]
    mask.mean = data["arr_1"]
    log.info(f"Loading eigenvectors:\n{mask.eigenVectors}")
    log.info(f"Loading mean:\n{mask.mean}")
    img = mask.getImage(name)
    dst = mask.reconstruct(img)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.subplot(122)
    if len(dst.shape) == 3:
        plt.imshow(dst[:, :, ::-1])
    else:
        plt.imshow(dst, cmap="gray")
    plt.show()


if __name__ == "__main__":
    test()
