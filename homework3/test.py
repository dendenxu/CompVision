#! python

# Example:
# python test.py -i image_0318.jpg -m model.color.npz -c builtin.json -o similar.png

from eigenface import *
import argparse
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def test(imgname, modelName, config, outputName):

    # instantiate new eigenface class
    mask = EigenFaceUtils()
    mask.loadConfig(config)
    # load previous eigenvectors/mean value
    mask.loadModel(modelName)
    txtname = f"{os.path.splitext(imgname)[0]}.txt"
    if os.path.isfile(txtname):
        log.info("Found text file")
        mask.updateEyeDictEntry(txtname)
    else:
        log.warning(f"Cannot find eye text file for test: {txtname}")
    log.info(f"Loading image: {imgname}")
    src = mask.getImage(imgname)
    dst, eigen, face, ori, dbImgName = mask.reconstruct(src)

    imgs = [src, dst, eigen, face, ori]
    imgBaseName = os.path.basename(imgname)
    dbImgBaseName = os.path.basename(dbImgName)
    msgs = [f"Original Test Image\n{imgBaseName}", f"Reconstructed Test Image\n{imgBaseName}", "Most Similar Eigen Face", f"Reconstructed Most Similar\nDatabase Image\n{dbImgBaseName}", f"Original Most Similar\nDatabase Image\n{dbImgBaseName}"]

    for i in range(len(imgs)):
        img = imgs[i]
        msg = msgs[i]
        offset = 20 / 512 * mask.h
        drawOffset = offset
        scale = 1 / 512 * mask.w
        thick = int(4 / 512 * mask.w)
        splitted = msg.split("\n")
        for msg in splitted:
            size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(img, msg, (int((mask.w-size[0][0])/2), int(drawOffset+size[0][1])), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
            offset = size[0][1]*1.5  # 1.5 line height
            drawOffset += offset

    w = mask.w
    h = mask.h
    if mask.isColor:
        canvas = np.zeros((h, 5*w, 3), dtype="uint8")
    else:
        canvas = np.zeros((h, 5*w), dtype="uint8")
    canvas[:, 0*w:1*w] = src
    canvas[:, 1*w:2*w] = dst
    canvas[:, 2*w:3*w] = eigen
    canvas[:, 3*w:4*w] = face
    canvas[:, 4*w:5*w] = ori

    if outputName is not None:
        log.info(f"Saving output to {outputName}")
        cv2.imwrite(outputName, canvas)
    else:
        log.warning(f"You didn't specify a output file name, the result WILL NOT BE SAVED\nIt's highly recommended to save the result with -o argument since OpenCV can't even draw large window properly...")

    if not mask.useHighgui:
        log.error("Only HighGUI of OpenCV is supported.\nOther implementation removed due to regulation.")
    else:

        window = "Original | EigenReconstruct | Most Similar"
        cv2.namedWindow(window)
        cv2.imshow(window, canvas)
        cv2.waitKey()
        cv2.destroyWindow(window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
We're assuming a <imageFileNameNoExt>.txt for eye position like
474 247 607 245
Comment line starting with # will be omitted
Or we'll use OpenCV's haarcascade detector to try locating eyes' positions

Note that if you want to save the recognition result
pass -o argument to specify the output file name
It's highly recommended to do so since OpenCV can't even draw large window properly...
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default="image_0318.jpg", help="The image we want to reconstruct and recognize on")
    parser.add_argument("-m", "--model", default="model.color.npz", help="The model trained with this eigenface utility")
    parser.add_argument("-c", "--config", default="builtin.json", help="The configuration file for the eigenface utility instance")
    parser.add_argument("-o", "--output", help="The output file to save the reconstruction/recognition result. If not specified, the program WILL NOT SAVE THE RESULT")

    args = parser.parse_args()
    test(args.input, args.model, args.config, args.output)
