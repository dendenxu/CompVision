import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
import logging
import coloredlogs
import argparse

# Setting up font for matplotlib
mpl.rc("font", family=["Josefin Sans", "Trebuchet MS", "Inconsolata"], weight="medium")

# Setting up logger for the project
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Adding argument parser
parser = argparse.ArgumentParser(description="""""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-d", "--directory", default="./birdseye", help="The folder containing all images to be used to generate birdseye view")
parser.add_argument("-c", "--columns", type=int, default=12, help="The width of the checkerboard. (Points to be detected along a horizontal line)")
parser.add_argument("-r", "--rows", type=int, default=12, help="The height of the checkerboard. (Points to be detected along a vertical line)")
parser.add_argument("-i", "--input_file", default="intrinsics.xml", help="The intrinsics.xml that file path(with file name) that contains camera calibration information: intrinsics and distortion coefficients")

# parse the arguments
args = parser.parse_args()
intrfile = args.input_file
path = args.directory
board_w = args.columns
board_h = args.rows
# compute things from the arguments
fpaths = [os.path.join(path, fname) for fname in os.listdir(path)]
board_n = board_w*board_h
board_sz = (board_w, board_h)

# read in camera calibration information:
# camera intrinsics and distortion coefficients
fs = cv2.FileStorage(intrfile, cv2.FILE_STORAGE_READ)
img_shp = tuple(map(int, (fs.getNode("image_width").real(), fs.getNode("image_height").real())))
intr = fs.getNode("camera_matrix").mat()
dist = fs.getNode("distortion_coefficients").mat()


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# object points to be used to generate birdseye view
objPts = np.zeros((4, 2), "float32")
objPts[0][0] = 0
objPts[0][1] = 0
objPts[1][0] = board_w - 1
objPts[1][1] = 0
objPts[2][0] = 0
objPts[2][1] = board_h - 1
objPts[3][0] = board_w - 1
objPts[3][1] = board_h - 1
# ! test code
# objPts *= 10

# image points
imgPts = np.zeros((4, 2), "float32")

for fpath in fpaths:
    img_dist = cv2.imread(fpath)
    img = cv2.undistort(img_dist, intr, dist)
    img_shp = img.shape[:2][::-1]  # OpenCV wants (width, height)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_sz, None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        break
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgPts[0] = corners[0]
    imgPts[1] = corners[board_w - 1]
    imgPts[2] = corners[(board_h - 1) * board_w]
    imgPts[3] = corners[(board_h - 1) * board_w + board_w - 1]

    cv2.circle(img, tuple(imgPts[0].astype(int).tolist()), 9, (255, 0, 0), 3)
    cv2.circle(img, tuple(imgPts[1].astype(int).tolist()), 9, (0, 255, 0), 3)
    cv2.circle(img, tuple(imgPts[2].astype(int).tolist()), 9, (0, 0, 255), 3)
    cv2.circle(img, tuple(imgPts[3].astype(int).tolist()), 9, (0, 255, 255), 3)

    cv2.drawChessboardCorners(img, board_sz, corners, found)
    H = cv2.getPerspectiveTransform(objPts, imgPts)
    Z = H[2, 2]
    S = 10.0
    quit = False
    shape = img.shape[:2][::-1]
    log.info("Press 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit")
    log.info(f"Getting shape: {shape} for image {fpath}")
    while True:
        # H = cv2.getPerspectiveTransform(objPts*S, imgPts)
        H[2, 2] = Z
        log.info(f"Getting H of {H}, Z of {Z}")
        # when the flag WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert and then put in the formula above instead of M. The function cannot operate in-place.
        scale = np.diag([1/S, 1/S, 1])
        birdseye = cv2.warpPerspective(
            img, np.matmul(H, scale), shape,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imshow("Rectified Img", img)
        cv2.imshow("Birdseye View", birdseye)
        k = cv2.waitKey(0) & 0xff
        log.info(f"Getting key: {chr(k)} or order {k}")
        if k == ord('u'):
            Z += 0.1
        if k == ord('d'):
            Z -= 0.1
        if k == ord('i'):
            S += 0.5
        if k == ord('o'):
            S -= 0.5
        if k == ord('n'):
            break
        if k == 27:
            quit = True
            cv2.destroyAllWindows()
            break
    if quit:
        break
