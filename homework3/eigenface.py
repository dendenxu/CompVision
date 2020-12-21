import cv2
import numpy as np
import logging
import coloredlogs
import random
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from tqdm import tqdm
import scipy.sparse.linalg as sla
import scipy.linalg as la
import sys
import json

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
    def __init__(self, width=128, height=128, left=np.array([48, 48]), right=np.array([80, 48]), isColor=False, nEigenFaces=None, targetPercentage=None, useBuiltin=False, useHighgui=True):
        # # at least one of them should be provided
        # assert nEigenFaces is not None or targetPercentage is not None
        self.width = width
        self.height = height
        self.left = left  # x, y
        self.right = right  # x, y
        self.face_cascade = None
        self.eye_cascade = None
        self.eyesDict = {}
        self.batch = None  # samples
        self.covar = None  # covariances
        self.eigenValues = None
        self.eigenVectors = None
        self.eigenFaces = None
        self.mean = None
        self.isColor = isColor
        self.nEigenFaces = nEigenFaces
        self.targetPercentage = targetPercentage
        self.useBuiltin = useBuiltin
        self.useHighgui = useHighgui

        # ! cannot be pickled, rememeber to delete after loading
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    def alignFace(self, face: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
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

        if scale > 1:
            M = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]]])
            face = cv2.warpAffine(face, M, (self.width, self.height))
            M = cv2.getRotationMatrix2D(tuple(maskCenter), angle, scale)
            face = cv2.warpAffine(face, M, (self.width, self.height))
        else:
            M = cv2.getRotationMatrix2D(tuple(faceCenter), angle, scale)
            face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
            M = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]]])
            face = cv2.warpAffine(face, M, (self.width, self.height))
        return face

    def detectFace(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = np.ndarray((0, 2))
        # assuming only one face here
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            new = self.eye_cascade.detectMultiScale(roi_gray)
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

    @property
    def shouldLen(self):
        return self.colorLen if self.isColor else self.grayLen

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
            # cannot find the already processed data in dict
            if self.isColor:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            return self.detectFace(gray)[1]  # we only need the eyes
        else:
            return self.eyesDict[name]

    def getImage(self, name, manual_check=False) -> np.ndarray:
        # the load the image accordingly
        if self.isColor:
            img = cv2.imread(name, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

        # try getting eye position
        eyes = self.getEyes(name, img)
        log.info(f"Getting eyes: {eyes}")
        if not len(eyes) == 2:
            log.warning(f"Cannot get two eyes from this image: {name}, {len(eyes)} eyes")
            raise EigenFaceException("Bad image")

        # align according to eye position
        dst = self.alignFace(img, eyes[0], eyes[1])

        # hist equalization
        if self.isColor:
            dst = self.equalizeHistColor(dst)
        else:
            dst = cv2.equalizeHist(dst)

        # should we check every image before/after loading?
        if manual_check:
            cv2.imshow(name, dst)
            cv2.waitKey()
            cv2.destroyWindow(name)
        return dst

    def getBatch(self, path="./", ext=".jpg", manual_check=False, append=False) -> np.ndarray:
        # adjust logging level to be quite or not
        prevLevel = coloredlogs.get_level()
        if not manual_check:
            coloredlogs.set_level("WARNING")

        names = os.listdir(path)
        names = [os.path.join(path, name) for name in names if name.endswith(ext)]
        if not append:
            if self.isColor:
                self.batch = np.ndarray((0, self.colorLen))  # assuming color
            else:
                self.batch = np.ndarray((0, self.grayLen))
        for name in tqdm(names, desc="Processing batch"):
            try:
                dst = self.getImage(name, manual_check)
                flat = dst.flatten()
                flat = np.reshape(flat, (1, len(flat)))
                self.batch = np.concatenate([self.batch, flat])
            except EigenFaceException as e:
                log.warning(e)

        coloredlogs.set_level(prevLevel)

        return self.batch

    def getDict(self, path="./", ext=".eye", manual_check=False) -> dict:
        prevLevel = coloredlogs.get_level()
        if not manual_check:
            coloredlogs.set_level("WARNING")

        names = os.listdir(path)
        names = [os.path.join(path, name) for name in names if name.endswith(ext)]
        log.info(f"Good names: {names}")
        for name in names:
            # iterate through all txt files
            self.getDictEntry(name)

        # restore the logging level
        coloredlogs.set_level(prevLevel)
        return self.eyesDict

    def getDictEntry(self, name):
        with open(name, "r") as f:
            lines = f.readlines()
            log.info(f"Processing: {name}")
            for line in lines:  # actually there should only be one line
                line = line.strip()  # get rid of starting/ending space \n
                # assuming # starting line to be comment
                if line.startswith("#"):  # get rid of comment file
                    log.info(f"Getting comment line: {line}")
                    continue
                coords = line.split()
                name = os.path.basename(name)  # get file name
                name = os.path.splitext(name)[0]  # without ext
                if len(coords) == 4:
                    self.eyesDict[name] = np.reshape(np.array(coords).astype(int), [2, 2])
                    order = np.argsort(self.eyesDict[name][:, 0])  # sort by first column, which is x
                    self.eyesDict[name] = self.eyesDict[name][order]
                else:
                    log.error(f"Wrong format for file: {name}, at line: {line}")

    def getCovarMatrix(self) -> np.ndarray:
        assert self.batch is not None and self.mean is not None
        log.info(f"Trying to compute the covariance matrix")
        # covariance matrix of all the pixel location: width * height * color
        self.covar = np.cov(np.transpose(self.batch-self.mean))  # subtract mean
        log.info(f"Getting covar of shape: {self.covar.shape}")
        log.info(f"Getting covariance matrix:\n{self.covar}")
        return self.covar

    def getEigen(self) -> np.ndarray:
        assert self.covar is not None

        if self.targetPercentage is not None:
            log.info(f"Begin computing all eigenvalues")
            self.eigenValues = la.eigvalsh(self.covar)
            self.eigenValues = np.sort(self.eigenValues)[::-1]  # this should be sorted
            log.info(f"Getting all eigenvalues:\n{self.eigenValues}\nof shape: {self.eigenValues.shape}")
            self.getnEigenFaces()

        log.info(f"Begin computing {self.nEigenFaces} eigenvalues/eigenvectors")
        self.eigenValues, self.eigenVectors = sla.eigen.eigsh(self.covar, k=self.nEigenFaces)
        log.info(f"Getting {self.nEigenFaces} eigenvalues and eigenvectors with shape {self.eigenVectors.shape}")

        # always needed right?
        self.eigenVectors = np.transpose(self.eigenVectors.astype("float64"))

        # ? probably not neccessary?
        # might already be sorted according to la.eigen.eigs' algorithm
        order = np.argsort(self.eigenValues)[::-1]
        self.eigenValues = self.eigenValues[order]
        self.eigenVectors = self.eigenVectors[order]

        log.info(f"Getting sorted eigenvalues:\n{self.eigenValues}\nof shape: {self.eigenValues.shape}")
        log.info(f"Getting sorted eigenvectors:\n{self.eigenVectors}\nof shape: {self.eigenVectors.shape}")

        return self.eigenValues, self.eigenVectors

    def reconstruct(self, img: np.ndarray) -> np.ndarray:
        assert self.eigenVectors is not None and self.mean is not None

        result = self.unflatten(self.mean)  # mean face
        flat = img.flatten().astype("float64")  # loaded image with double type
        flat = np.expand_dims(flat, 0)  # viewed as 1 * (width * height * color)
        flat -= self.mean  # flatten subtracted with mean face
        flat = np.transpose(flat)  # (width * height * color) * 1
        log.info(f"Shape of eigenvectors and flat: {self.eigenVectors.shape}, {flat.shape}")

        # nEigenFace *(width * height * color) matmal (width * height * color) * 1
        weights = np.matmul(self.eigenVectors, flat)  # new data, nEigenFace * 1

        # luckily, transpose of eigenvector is its inversion
        # Eigenvectors of real symmetric matrices are orthogonal
        # ! the magic happens here
        # data has been lost because nEigenFaces is much smaller than the image dimension span
        # which is width * height * color
        # but because we're using PCA (principal components), most of the information will still be retained
        flat = np.matmul(np.transpose(self.eigenVectors), weights)  # restored
        log.info(f"Shape of flat: {flat.shape}")
        flat = np.transpose(flat)
        result += self.unflatten(flat)
        return result

    def getMean(self):
        assert self.batch is not None
        # get the mean values of all the vectorized faces
        self.mean = np.reshape(np.mean(self.batch, 0), (1, -1))
        log.info(f"Getting mean vectorized face: {self.mean} with shape: {self.mean.shape}")
        return self.mean

    def unflatten(self, flat: np.ndarray) -> np.ndarray:
        # rubust method for reverting a flat matrix
        if len(flat.shape) == 2:
            length = flat.shape[1]
        else:
            length = flat.shape[0]
        if length == self.grayLen:
            if self.isColor:
                log.warning("You're reshaping a grayscale image when color is wanted")
            return np.reshape(flat, (self.height, self.width))
        elif length == self.colorLen:
            if self.isColor:
                log.warning("You're reshaping a color image when grayscale is wanted")
            return np.reshape(flat, (self.height, self.width, 3))
        else:
            raise EigenFaceException(f"Unsupported flat array of length: {length}, should provide {self.grayLen} or {self.colorLen}")

    def uint8unflatten(self, flat):
        # for displaying
        img = self.unflatten(flat)
        return img.astype("uint8")

    def getnEigenFaces(self):
        assert self.eigenValues is not None
        # get energy
        self.nEigenFaces = len(self.eigenValues)
        targetValue = np.sum(self.eigenValues) * self.targetPercentage
        accumulation = 0
        for index, value in enumerate(self.eigenValues):
            accumulation += value
            if accumulation > targetValue:
                self.nEigenFaces = index + 1
                log.info(f"For a energy percentage of {self.targetPercentage}, we need {self.nEigenFaces} vectors from {len(self.eigenValues)}")
                break  # current index should be nEigenFaces

    def train(self, path, imgext, txtext, modelName="model.npz"):
        self.getDict(path, txtext)
        self.getBatch(path, imgext)
        if self.useBuiltin:
            if self.targetPercentage is not None:
                log.info(f"Beginning builtin PCACompute2 for all eigenvalues/eigenvectors")
                # ! this is bad, we'll have to compute all eigenvalues/eigenvectors to determine energy percentage
                self.mean, self.eigenVectors, self.eigenValues = cv2.PCACompute2(self.batch, None)
                log.info(f"Getting eigenvalues/eigenvectors: {self.eigenValues}, {self.eigenVectors}")
                self.getnEigenFaces()
                # ! dangerous, losing smaller eigenvectors (eigenvalues is small)
                self.eigenVectors = self.eigenVectors[0:self.nEigenFaces]
            else:
                log.info(f"Beginning builtin PCACompute for {self.nEigenFaces} eigenvalues/eigenvectors")
                self.mean, self.eigenVectors = cv2.PCACompute(self.batch, None, maxComponents=self.nEigenFaces)
            log.info(f"Getting mean vectorized face: {self.mean} with shape: {self.mean.shape}")
            log.info(f"Getting sorted eigenvectors:\n{self.eigenVectors}\nof shape: {self.eigenVectors.shape}")
        else:
            self.getMean()
            self.getCovarMatrix()
            self.getEigen()
        self.saveModel(modelName)

    def loadModel(self, modelName):
        # load previous eigenvectors/mean value
        log.info(f"Loading model: {modelName}")
        data = np.load(modelName)
        self.eigenVectors = data["arr_0"]
        self.mean = data["arr_1"]
        log.info(f"Getting mean vectorized face: {self.mean} with shape: {self.mean.shape}")
        log.info(f"Getting sorted eigenvectors:\n{self.eigenVectors}\nof shape: {self.eigenVectors.shape}")

    def saveModel(self, modelName):
        log.info(f"Saving model: {modelName}")
        np.savez_compressed(modelName, self.eigenVectors, self.mean)
        log.info(f"Model: {modelName} saved")

    # ! unused
    def getEigenFaces(self) -> np.ndarray:
        assert self.eigenVectors is not None
        log.info(f"Computing eigenfaces")
        self.eigenFaces = np.array([self.unflatten(vector) for vector in self.eigenVectors])
        log.info(f"Getting eigenfaces of shape {self.eigenFaces.shape}")
        return self.eigenFaces

    @staticmethod
    def randcolor():
        '''
        Generate a random color, as list
        '''
        return [random.randint(0, 256) for _ in range(3)]

    def getCovarMatrixSlow(self) -> np.ndarray:
        assert self.batch is not None, "Should get sample batch before computing covariance matrix"
        nSamples = self.batch.shape[0]
        self.covar = np.zeros((nSamples, nSamples))
        for k in tqdm(range(nSamples**2), "Getting covariance matrix"):
            i = k // nSamples
            j = k % nSamples
            linei = self.batch[i]
            linej = self.batch[j]
            # naive!!!
            if self.covar[j][i] != 0:
                self.covar[i][j] = self.covar[j][i]
            else:
                self.covar[i][j] = self.getCovar(linei, linej)

    @staticmethod
    def getCovar(linei, linej) -> np.ndarray:
        # naive
        meani = np.mean(linei)
        meanj = np.mean(linej)
        unbiasedi = linei - meani
        unbiasedj = linej - meanj
        multi = np.dot(unbiasedi, unbiasedj)
        multi /= len(linei) - 1
        return multi

    def loadConfig(self, filename):
        with open(filename, "rb") as f:
            data = json.load(f)
        log.info(f"Loading configuration from {filename}, with content: {data}")
        self.width = data["width"]
        self.height = data["height"]
        self.left = np.array(data["left"])
        self.right = np.array(data["right"])
        self.isColor = data["isColor"]
        self.nEigenFaces = data["nEigenFaces"]

        # setting this to null will reduce computation significantly
        # since we'll only have to compute nEigenFaces eigenvalues/eigenvectors
        # setting to a value would disable nEigenFaces parameter, computing from target information retain rate "targetPercentage"
        self.targetPercentage = data["targetPercentage"]

        # whether we should use cv2.PCACompute/cv2.PCACompute2 or our own PCA computation
        # note the builtin methods are much faster and memory efficient if we're only requiring
        # a small subset of all eigenvectors (don't have to allocate the covariance matrix)
        # enabling us to do large scale computation, even with colored image of 512 * 512
        self.useBuiltin = data["useBuiltin"]

        # whether we should use matplotlib to draw results or HIGHGUI of opencv
        # I think HIGHGUI sucks at basic figure management, I'd prefer matplotlib for figure drawing
        # but if we're looking for a more general solution for GUI, it is a choice
        self.useHighgui = data["useHighgui"]

    def saveConfig(self, filename):
        data = {}
        data["width"] = self.width
        data["height"] = self.height
        data["left"] = self.left.tolist()
        data["right"] = self.right.tolist()
        data["isColor"] = self.isColor
        data["nEigenFaces"] = self.nEigenFaces
        data["targetPercentage"] = self.targetPercentage
        data["useBuiltin"] = self.useBuiltin
        data["useHighgui"] = self.useHighgui

        log.info(f"Dumping configuration to {filename}, with content: {data}")
        with open(filename, "wb") as f:
            json.dump(data, f)
