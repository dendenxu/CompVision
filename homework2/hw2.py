#!python
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys  # for commandline arguments
import cv2
import random
import logging
import coloredlogs
import numpy as np
# from matplotlib import rcParams
# # rcParams['font.family'] = 'monospace'
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.monospace'] = ['Inconsolata', 'Fira Code', 'PT Mono']
# # rcParams['font.stretch'] = 'condensed'
# rcParams['font.weight'] = "light"
# rcParams['font.sans-serif'] = ['Josefin Sans']
mpl.rc("font", family=["Josefin Sans", "Trebuchet MS", "Inconsolata"])

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def randcolor():
    return [random.randint(0, 256) for i in range(3)]


def getImg(imgName):
    img = cv2.imread(imgName)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE) # assuming the first user defined arg as img name
    edge = cv2.Canny(gray, 100, 200)  # using Canny to get edge
    return img, rgb, gray, edge


def getFeatures(edge):
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [None]*len(contours)
    ellipses = [None]*len(contours)
    for index, contour in enumerate(contours):
        rects[index] = cv2.minAreaRect(contour)
        if contour.shape[0] > 5:
            ellipses[index] = cv2.fitEllipse(contour)
    return contours, rects, ellipses


def render(shape, contours, rects, ellipses):
    # plt.figure()
    drawing = np.zeros(shape, dtype='uint8')
    log.info(f"Drawing on shape: {shape}")
    for index, contour in enumerate(contours):
        color = randcolor()
        log.debug(f"Getting random color: {color}")
        cv2.drawContours(drawing, contours, index, color, 1, cv2.LINE_AA)
        if contour.shape[0] > 5:
            cv2.ellipse(drawing, ellipses[index], color, 3, cv2.LINE_AA)

        box = cv2.boxPoints(rects[index])
        box = box.astype(int)
        if box.shape[0] > 0:
            cv2.drawContours(drawing, [box], 0, color, 2, cv2.LINE_AA)

        # plt.imshow(drawing)
        # plt.show()

    return drawing


def main():
    if len(sys.argv) > 1:
        imgName = sys.argv[1]
    else:
        imgName = "camaro.jpg"

    if not os.path.exists(imgName):
        log.error(f"Cannot find the file specified: {imgName}")
        log.info(f"This program fits ellipses to an image, usage:\npython hw2.py <image name>")
        return 1
    
    img, rgb, gray, edge = getImg(imgName)
    contours, rects, ellipses = getFeatures(edge)
    drawing = render(img.shape, contours, rects, ellipses)

    # plt.figure(figsize=(16, 9))
    plt.figure("Fitting")
    plt.suptitle("Fitting ellipses and finding min-area rectangles", fontweight="bold")
    plt.subplot(221)
    plt.title("Original", fontweight="bold")
    plt.imshow(rgb)
    plt.subplot(222)
    plt.title("Grayscale", fontweight="bold")
    plt.imshow(gray, cmap="gray")
    plt.subplot(223)
    plt.title("Canny Edge", fontweight="bold")
    plt.imshow(edge, cmap="gray")
    plt.subplot(224)
    plt.title("Contour & Ellipses & Rects", fontweight="bold")
    plt.imshow(drawing)
    plt.show()


if __name__ == "__main__":
    main()
