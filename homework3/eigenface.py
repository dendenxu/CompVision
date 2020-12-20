import cv2
import numpy as np
import logging
import coloredlogs
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.linalg as la

log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


class EigenFaceException(Exception):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class EigenFace:
    # desired face mask values to be used
    def __init__(self, eyesDict=None):
        self.width = 512
        self.height = 512
        self.left = np.array([188, 188])  # x, y
        self.right = np.array([324, 188])  # x, y
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.eyesDict = {} if eyesDict is None else eyesDict
        self.batch = None  # samples
        self.covar = None  # covariances
        self.eigenValues = None
        self.eigenVectors = None

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
        name = os.path.basename(name)  # get file name
        name = os.path.splitext(name)[0]  # without ext
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
            log.warning(f"Cannot get two eyes from this image: {name}, {len(eyes)} eyes")
            raise EigenFaceException("Bad image")
        dst = self.alignFace(img, eyes[0], eyes[1])
        dst = EigenFace.equalizeHistColor(dst)
        if manual_check:
            cv2.imshow(name, dst)
            cv2.waitKey()
            cv2.destroyWindow(name)
        return dst

    def getBatch(self, path="./", ext=".jpg", manual_check=False, append=False) -> np.ndarray:
        prevLevel = coloredlogs.get_level()
        if not manual_check:
            coloredlogs.set_level("WARNING")

        names = os.listdir(path)
        names = [os.path.join(path, name) for name in names if name.endswith(ext)]
        if not append:
            self.batch = np.ndarray((0, self.colorLen))  # assuming color
        for name in tqdm(names, desc="Processing batch"):
            try:
                dst = self.getImage(name, manual_check)
                flat = dst.flatten()
                flat = np.reshape(flat, (1, len(flat)))
                self.batch = np.concatenate([self.batch, flat])
            except EigenFaceException as e:
                log.error(e)

        coloredlogs.set_level(prevLevel)

        return self.batch

    def getDict(self, path="./", ext=".eye", quite=True) -> dict:

        prevLevel = coloredlogs.get_level()
        if quite:
            coloredlogs.set_level("WARNING")

        names = os.listdir(path)
        names = [os.path.join(path, name) for name in names if name.endswith(ext)]
        # assuming # starting line to be comment
        log.info(f"Good names: {names}")
        for name in names:
            with open(name, "r") as f:
                lines = f.readlines()
                log.info(f"Processing: {name}")
                for line in lines:  # actually there should only be one line
                    line = line.strip()  # get rid of starting/ending space \n
                    if not line.startswith("#"):  # get rid of comment file
                        coords = line.split()
                        name = os.path.basename(name)  # get file name
                        name = os.path.splitext(name)[0]  # without ext
                        if len(coords) == 4:
                            self.eyesDict[name] = np.reshape(np.array(coords).astype(int), [2, 2])
                            order = np.argsort(self.eyesDict[name][:, 0])  # sort by first column, which is x
                            self.eyesDict[name] = self.eyesDict[name][order]
                        else:
                            log.error(f"Wrong format for file: {name}, at line: {line}")
                    else:
                        log.info(f"Getting comment line: {line}")
        coloredlogs.set_level(prevLevel)
        return self.eyesDict

    def getCovarMatrix(self) -> np.ndarray:
        self.covar = np.cov(self.batch)
        return self.covar

    def getEigen(self) -> np.ndarray:
        self.eigenValues, self.eigenVectors = la.eig(self.covar)
        order = np.argsort(self.eigenValues)[::-1]
        self.eigenValues = self.eigenValues[order]
        self.eigenVectors = self.eigenVectors[order]
        return self.eigenValues, self.eigenVectors

    # def getCovarMatrix(self) -> np.ndarray:
    #     assert(self.batch is not None, "Should get sample batch before computing covariance matrix")
    #     nSamples = self.batch.shape[0]
    #     self.covar = np.zeros((nSamples, nSamples))
    #     for k in tqdm(range(nSamples**2), "Getting covariance matrix"):
    #         i = k // nSamples
    #         j = k % nSamples
    #         linei = self.batch[i]
    #         linej = self.batch[j]
    #         # naive!!!
    #         if self.covar[j][i] != 0:
    #             self.covar[i][j] = self.covar[j][i]
    #         else:
    #             self.covar[i][j] = self.getCovar(linei, linej)

    # @staticmethod
    # def getCovar(linei, linej) -> np.ndarray:
    #     # naive
    #     meani = np.mean(linei)
    #     meanj = np.mean(linej)
    #     unbiasedi = linei - meani
    #     unbiasedj = linej - meanj
    #     multi = np.dot(unbiasedi, unbiasedj)
    #     multi /= len(linei) - 1
    #     return multi

    def getMean(self):
        return np.reshape(np.mean(self.batch, 0), (1, -1))

    def unflatten(self, flat: np.ndarray) -> np.ndarray:
        length = flat.shape[1]
        if length == self.grayLen:
            return np.reshape(flat, (self.height, self.width))
        elif length == self.colorLen:
            return np.reshape(flat, (self.height, self.width, 3))
        else:
            raise EigenFaceException(f"Unsupported flat array of length: {length}, should provide {self.grayLen} ro {self.colorLen}")

    def uint8unflatten(self, flat):
        img = self.unflatten(flat)
        return img.astype("uint8")


def main():
    mask = EigenFace()
    # mask.getDict("./BioFaceDatabase/BioID-FaceDatabase-V1.2", ".eye")
    # batch = mask.getBatch("./BioFaceDatabase/BioID-FaceDatabase-V1.2", ".pgm")
    batch = mask.getBatch("./smallSet", ".jpg")
    mean = mask.getMean()
    img = mask.uint8unflatten(mean)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    covar = mask.getCovarMatrix()
    log.info(f"Getting covariance matrix:\n{covar}")
    values, vectors = mask.getEigen()
    log.info(f"Getting sorted eigenvalues:\n{values}")
    log.info(f"Getting sorted eigenvectors:\n{vectors}")
