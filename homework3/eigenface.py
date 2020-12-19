import cv2
import numpy as np
import logging
import coloredlogs
import random

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class FaceMask:
    # desired face mask values to be used
    def __init__(self):
        self.width = 512
        self.height = 512
        self.left = np.array([188, 188])
        self.right = np.array([188, 324])
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    def alignFace(mask, face: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # M = cv2.getAffineTransform(np.array([left, right]), np.array([mask.left, mask.right]))
        # dst = cv2.warpAffine(face, M, (mask.width, mask.height))
        # return dst
        faceVect = left - right
        maskVect = mask.left - mask.right
        log.info(f"Getting faceVect: {faceVect} and maskVect: {maskVect}")
        faceNorm = np.linalg.norm(faceVect)
        maskNorm = np.linalg.norm(maskVect)
        log.info(f"Getting faceNorm: {faceNorm} and maskNorm: {maskNorm}")
        scale = maskNorm / faceNorm
        log.info(f"Should scale the image to: {faceNorm}")
        faceAngle = np.degrees(np.arctan2(*faceVect))
        maskAngle = np.degrees(np.arctan2(*maskVect))
        angle = maskAngle - faceAngle
        log.info(f"Should rotate the image: {angle}degrees")
        faceCenter = (left+right)/2
        maskCenter = (mask.left+mask.right) / 2
        translation = maskCenter - faceCenter
        M = np.array([[1, 0, translation[0]],
                      [0, 1, translation[1]]])
        face = cv2.warpAffine(face, M, [mask.width, mask.height])
        M = cv2.getRotationMatrix2D(faceCenter, angle, scale)
        # Apply the translation to the warped image
        # M[0][2] += translation[0]
        # M[1][2] += translation[1]
        face = cv2.warpAffine(face, M, [mask.width, mask.height])

    def detectFace(mask, gray: np.ndarray) -> np.ndarray:
        faces = mask.face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = np.ndarray((0, 4))
        # assuming only one face here
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            new = mask.eye_cascade.detectMultiScale(roi_gray)
            for eye in new:
                eye[0] += x
                eye[1] += y
            eyes = np.concatenate([eyes, new])
        return faces, eyes


log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class FaceMask:
    # desired face mask values to be used
    def __init__(self):
        self.width = 512
        self.height = 512
        self.left = np.array([188, 188])  # x, y
        self.right = np.array([324, 188])  # x, y
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    def alignFace(mask, face: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # M = cv2.getAffineTransform(np.array([left, right]), np.array([mask.left, mask.right]))
        # dst = cv2.warpAffine(face, M, (mask.width, mask.height))
        # return dst
        faceVect = left - right
        maskVect = mask.left - mask.right
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
        maskCenter = (mask.left+mask.right) / 2
        log.info(f"Getting faceCenter: {faceCenter} and maskCenter: {maskCenter}")
        translation = maskCenter - faceCenter
        log.info(f"Should translate the image using: {translation}")
        M = np.array([[1, 0, translation[0]],
                      [0, 1, translation[1]]])
        face = cv2.warpAffine(face, M, (mask.width, mask.height))
        M = cv2.getRotationMatrix2D(tuple(maskCenter), angle, scale)
        # Apply the translation to the warped image
        # M[0][2] += translation[0]
        # M[1][2] += translation[1]
        face = cv2.warpAffine(face, M, (mask.width, mask.height))
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


def randcolor():
    '''
    Generate a random color, as list
    '''
    return [random.randint(0, 256) for i in range(3)]


def main():
    # img = cv2.imread("test.tiff")
    img = cv2.imread("image_0001.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = FaceMask()
    faces, eyes = mask.detectFace(gray)
    log.info(f"Getting {faces} and {eyes}")
    # assuming the first eye
    dst = mask.alignFace(gray, eyes[0], eyes[1])
    color = randcolor()
    # cv2.rectangle(dst, mask.left - [10, 10], mask.left + [10, 10], color, 2, cv2.LINE_AA)
    # cv2.rectangle(dst, mask.right - [10, 10], mask.right + [10, 10], color, 2, cv2.LINE_AA)
    cv2.imshow("Aligned", dst)
    cv2.waitKey()
    cv2.destroyWindow("Aligned")
