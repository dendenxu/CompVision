import cv2
import numpy as np
import logging
import coloredlogs
import random
import os

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class FaceMask:
    # desired face mask values to be used
    def __init__(self, eyesDict=None):
        self.width = 512
        self.height = 512
        self.left = np.array([188, 188])  # x, y
        self.right = np.array([324, 188])  # x, y
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.eyesDict = {} if eyesDict is None else eyesDict

    def alignFace(self, face: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # M = cv2.getAffineTransform(np.array([left, right]), np.array([mask.left, mask.right]))
        # dst = cv2.warpAffine(face, M, (mask.width, mask.height))
        # return dst
        faceVect = left - right
        maskVect = self.left - self.right
        log.info(f"Getting faceVect: {faceVect} and maskVect: {maskVect}")
        faceNorm = np.linalg.norm(faceVect)
        maskNorm = np.linalg.norm(maskVect)
        log.info(f"Getting faceNorm: {faceNorm} and maskNorm: {maskNorm}")
        scale = maskNorm / faceNorm
        log.info(f"Should scale the image to: {scale}")
        faceAngle = np.degrees(np.arctan2(*faceVect))
        maskAngle = np.degrees(np.arctan2(*maskVect))
        angle = maskAngle - faceAngle
        log.info(f"Should rotate the image: {maskAngle} - {faceAngle} = {angle} degrees")
        faceCenter = (left+right)/2
        maskCenter = (self.left+self.right) / 2
        log.info(f"Getting faceCenter: {faceCenter} and maskCenter: {maskCenter}")
        translation = maskCenter - faceCenter
        log.info(f"Should translate the image using: {translation}")
        M = np.array([[1, 0, translation[0]],
                      [0, 1, translation[1]]])
        face = cv2.warpAffine(face, M, (self.width, self.height))
        M = cv2.getRotationMatrix2D(tuple(maskCenter), angle, scale)
        # Apply the translation to the warped image
        # M[0][2] += translation[0]
        # M[1][2] += translation[1]
        face = cv2.warpAffine(face, M, (self.width, self.height))
        return face

    def detectFace(mask, gray: np.ndarray) -> np.ndarray:
        faces = mask.face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = np.ndarray((0, 2))
        # assuming only one face here
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            new = mask.eye_cascade.detectMultiScale(roi_gray)
            for (dx, dy, dw, dh) in new:
                eyes = np.concatenate([eyes, np.array([[dx+dw/2+x, dy+dh/2+y]])])
        order = np.argsort(eyes[:, 0])  # sort by first column, which is x
        eyes = eyes[order]
        return faces, eyes

    @property
    def grayLen(self):
        return self.width*self.height

    @property
    def colorLen(self):
        return self.grayLen*3

    @staticmethod
    def randcolor():
        '''
        Generate a random color, as list
        '''
        return [random.randint(0, 256) for i in range(3)]

    @staticmethod
    def equalizeHistColor(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        log.info(f"Getting # of channels: {len(channels)}")
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
        return img

    def getEyes(self, name, img=None):
        name = os.path.basename(name)
        if not name in self.eyesDict:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self.detectFace(gray)[1]  # we only need the face
        else:
            return self.eyesDict[name]

    def getImage(self, name, manual_check=False) -> np.ndarray:
        img = cv2.imread(name)
        # _, eyes = self.detectFace(gray)
        eyes = self.getEyes(name, img)
        log.info(f"Getting eyes: {eyes}")
        if not len(eyes) == 2:
            log.warning(f"Cannot get enough information about this image: {name}")
            return np.ndarray((0, 0))
        dst = self.alignFace(img, eyes[0], eyes[1])
        dst = FaceMask.equalizeHistColor(dst)
        if manual_check:
            cv2.imshow(name, dst)
            cv2.waitKey()
            cv2.destroyWindow(name)
        return dst

    def getBatch(self, path="./", ext=".jpg", manual_check=False) -> np.ndarray:
        names = os.listdir(path)
        names = [os.path.join(path, name) for name in names if name.endswith(ext)]
        batch = np.ndarray((0, self.colorLen))  # assuming color
        for name in names:
            dst = self.getImage(name, manual_check)
            flat = dst.flatten()
            if len(flat) != self.colorLen:
                log.error(f"This image's length {len(flat)} doesn't match up with the mask ({self.colorLen}), single channel? Or grayscale?")
                continue
            flat = np.reshape(flat, (1, len(flat)))
            batch = np.concatenate([batch, flat])

        return batch


def main():
    # # img = cv2.imread("test.tiff")
    # img = cv2.imread("image_0001.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask = FaceMask()
    # faces, eyes = mask.detectFace(gray)
    # log.info(f"Getting {faces} and {eyes}")
    # # assuming the first eye
    # dst = mask.alignFace(img, eyes[0], eyes[1])
    # dst = FaceMask.equalizeHistColor(dst)
    # # color = randcolor()
    # # cv2.rectangle(dst, mask.left - [10, 10], mask.left + [10, 10], color, 2, cv2.LINE_AA)
    # # cv2.rectangle(dst, mask.right - [10, 10], mask.right + [10, 10], color, 2, cv2.LINE_AA)
    # cv2.imshow("Aligned", dst)
    # cv2.waitKey()
    # cv2.destroyWindow("Aligned")
    mask = FaceMask()
    batch = mask.getBatch("./Caltec Database -faces/", ".jpg")
