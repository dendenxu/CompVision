import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib as mpl
import logging
import coloredlogs

# Setting up font for matplotlib
mpl.rc("font", family=["Josefin Sans", "Trebuchet MS", "Inconsolata"], weight="medium")

# Setting up logger for the project
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# ! delete me
sys.argv.append("./intrinsics.xml")
sys.argv.append("./birdseye")
sys.argv.append("12")
sys.argv.append("12")

intrfile = sys.argv[1]
path = sys.argv[2]
board_w = int(sys.argv[3])
board_h = int(sys.argv[4])

fs = cv2.FileStorage(intrfile, cv2.FILE_STORAGE_READ)
img_shp = tuple(map(int, (fs.getNode("image_width").real(), fs.getNode("image_height").real())))
intr = fs.getNode("camera_matrix").mat()
dist = fs.getNode("distortion_coefficients").mat()

fpaths = [os.path.join(path, fname) for fname in os.listdir(path)]
board_n = board_w*board_h
board_sz = (board_w, board_h)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objPts = np.zeros((4, 2), "float32")
objPts[0][0] = 0
objPts[0][1] = 0
objPts[1][0] = board_w - 1
objPts[1][1] = 0
objPts[2][0] = 0
objPts[2][1] = board_h - 1
objPts[3][0] = board_w - 1
objPts[3][1] = board_h - 1

imgPts = np.zeros((4, 2), "float32")

for fpath in fpaths:
    img_dist = cv2.imread(fpath)
    img = cv2.undistort(img_dist, intr, dist)
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
    log.info("Press 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit")
    Z = 15
    quit = False
    shape = img.shape[:2][::-1]
    log.info(f"Getting shape: {shape} for image {fpath}")
    while True:
        H[2, 2] = Z
        birdseye = cv2.warpPerspective(
            img, H, shape,
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imshow("Rectified Img", img)
        cv2.imshow("Birdseye View", birdseye)
        k = cv2.waitKey(0) & 0xff
        if k == ord('u'):
            Z += 0.5
        if k == ord('d'):
            Z -= 0.5
        if k == ord('n'):
            break
        if k == 27:
            quit = True
            cv2.destroyAllWindows()
            break
    if quit:
        break
