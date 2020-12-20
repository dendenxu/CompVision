from eigenface import *


def main():
    path = "./smallSet"
    imgext = ".jpg"
    txtext = ".txt"
    if len(sys.argv) > 1:
        path = sys.argv[1]  # the first arg should be the path
    if len(sys.argv) > 2:
        imgext = sys.argv[2]  # the second arg should be image extension
    if len(sys.argv) > 3:
        txtext = sys.argv[3]  # the third should be the eyes position's text file's ext
    mask = EigenFace()
    eyes = mask.getDict(path, txtext)
    batch = mask.getBatch(path, imgext)
    mean = mask.getMean()
    img = mask.uint8unflatten(mean)
    if len(img.shape) == 3:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img, cmap="gray")
    plt.show()
    covar = mask.getCovarMatrix()
    log.info(f"Getting covariance matrix:\n{covar}")
    values, vectors = mask.getEigen()
    log.info(f"Getting sorted eigenvalues:\n{values}")
    log.info(f"Getting sorted eigenvectors:\n{vectors}")
    faces = mask.getEigenFaces()
    # with open("model", "wb") as f:
    #     mask.face_cascade = None
    #     mask.eye_cascade = None
    #     f.write(compress_pickle.dumps(mask, compression="gzip"))
    np.savez_compressed("model.npz", mask.eigenVectors, mask.mean)


if __name__ == "__main__":
    main()
