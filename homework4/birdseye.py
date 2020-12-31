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

# initializating stuff
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # criteria in subPix corner refinemnet
# the iteration of cornerSubPix stops when criteria.maxIter is reached or the corner position moves less than epsilon

objPts = np.zeros((4, 2), "float32")  # object points to be used to generate birdseye view
# currently using the borad's default coordinates
objPts[0][0] = 0
objPts[0][1] = 0
objPts[1][0] = board_w - 1
objPts[1][1] = 0
objPts[2][0] = 0
objPts[2][1] = board_h - 1
objPts[3][0] = board_w - 1
objPts[3][1] = board_h - 1

# image points
imgPts = np.zeros((4, 2), "float32")

for fpath in fpaths:
    img_dist = cv2.imread(fpath)
    if img_dist is None:
        log.warning(f"Cannot read image {fpath}, is this really a image?")
        break
    img = cv2.undistort(img_dist, intr, dist)  # do an undistortion using the parameters obtained from calibrate.py
    img_shp = img.shape[:2][::-1]  # OpenCV wants (width, height)
    found, corners = cv2.findChessboardCorners(img, board_sz, None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)  # approximate checker board location
    if not found:
        log.warning(f"Cannot find checker board from image {fpath}, is there really a checker board? Do your width/height match?")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # OpenCV wants gray image for subPix detection
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # refine checker board location

    # points on the image plane
    imgPts[0] = corners[0]
    imgPts[1] = corners[board_w - 1]
    imgPts[2] = corners[(board_h - 1) * board_w]
    imgPts[3] = corners[(board_h - 1) * board_w + board_w - 1]

    # mark the image points to be used to generate birdseye view
    cv2.circle(img, tuple(imgPts[0].astype(int).tolist()), 9, (255, 0, 0), 3)
    cv2.circle(img, tuple(imgPts[1].astype(int).tolist()), 9, (0, 255, 0), 3)
    cv2.circle(img, tuple(imgPts[2].astype(int).tolist()), 9, (0, 0, 255), 3)
    cv2.circle(img, tuple(imgPts[3].astype(int).tolist()), 9, (0, 255, 255), 3)

    # draw the checker board to amuse the user
    cv2.drawChessboardCorners(img, board_sz, corners, found)
    H = cv2.getPerspectiveTransform(objPts, imgPts)  # transform matrix, from objPts to imgPts (map points in objPts to imgPts)
    Z = H[2, 2]  # Z value, view height
    S = 10.0  # Scale value
    quit = False  # should we quit?
    shape = img.shape[:2][::-1]  # img shape
    log.info("Press 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit, 'n' for the next image and 'i' to zoom in, 'o' to zoom out")
    log.info(f"Getting shape: {shape} for image {fpath}")
    while True:
        H[2, 2] = Z
        log.info(f"Getting H of {H}, Z of {Z}")
        # when the flag WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert and then put in the formula above instead of M. The function cannot operate in-place.
        scale = np.diag([1/S, 1/S, 1])  # scale matrix, scale down the matrix to make the result scale up

        # dst(x, y) are defined by src(homo(matmul(H, (x, y, 1))))
        # so the transformation actually acts on the coordinates of the dstination image
        # thus OpenCV would firstly apply a inversion by default
        birdseye = cv2.warpPerspective(
            img, np.matmul(H, scale), shape,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,  # don't do matrix inverse before transform
            borderMode=cv2.BORDER_CONSTANT  # use constant value to fill the border
        )

        # give the user two image to savor on
        cv2.imshow("Rectified Img", img)
        cv2.imshow("Birdseye View", birdseye)

        k = cv2.waitKey(0) & 0xff
        log.info(f"Getting key: {chr(k)} or order {k}")

        # update view height
        if k == ord('u'):
            Z += 0.1
        if k == ord('d'):
            Z -= 0.1

        # update scale factor
        if k == ord('i'):
            S += 0.5
        if k == ord('o'):
            S -= 0.5

        # next image or
        # just quit already...
        if k == ord('n'):
            break
        if k == 27:
            quit = True
            cv2.destroyAllWindows()  # clean up before exiting
            break
    if quit:
        break
