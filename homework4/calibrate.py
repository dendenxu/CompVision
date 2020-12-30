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
sys.argv.append("./calibration")
sys.argv.append("12")
sys.argv.append("12")
sys.argv.append("True")

path = sys.argv[1]
fpaths = [os.path.join(path, fname) for fname in os.listdir(path)]
board_w = int(sys.argv[2])
board_h = int(sys.argv[3])
board_n = board_w*board_h
board_sz = (board_w, board_h)
show_imgs = sys.argv[4] == "True"


img_pts = []
obj_pts = []
img_shp = ()

objp = np.zeros((board_n, 3), "float32")
objp[:, :2] = np.mgrid[0:board_h, 0:board_w].T.reshape(-1, 2)

imgs = []

for fpath in fpaths:
    img = cv2.imread(fpath)
    img_shp = img.shape[:2][::-1]
    found, corners = cv2.findChessboardCorners(img, board_sz)
    if found:
        imgs.append(img)
        img_pts.append(corners)
        obj_pts.append(objp)

obj_pts = np.array(obj_pts).astype("float32")

err, intr, dist, rota, tran = cv2.calibrateCamera(obj_pts, img_pts, img_shp, None, None, flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT)
fs = cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_WRITE)
fs.write("image_width", img_shp[0])
fs.write("image_height", img_shp[1])
fs.write("camera_matrix", intr)
fs.write("distortion_coefficients", dist)
fs.release()

if show_imgs:
    # mapx, mapy = cv2.initUndistortRectifyMap(intr, dist, None, intr, img_shp, cv2.CV_16SC2)

    for i in range(len(imgs)):
        img = imgs[i]
        corners = img_pts[i]
        cv2.drawChessboardCorners(img, board_sz, corners, True)
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        dst = cv2.undistort(img, intr, dist)
        plt.subplot(121)
        plt.imshow(img)
        plt.title("Original")
        plt.subplot(122)
        plt.imshow(dst)
        plt.title("Rectified")
        plt.show()

