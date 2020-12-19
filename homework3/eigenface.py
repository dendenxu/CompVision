import cv2
import numpy as np

# desired face mask values to be used


class FaceMask:
    def __init__(self):
        self.width = 512
        self.height = 512
        self.left = np.array([188, 188])
        self.right = np.array([188, 324])


def alignFace(face: np.ndarray, left: np.ndarray, right: np.ndarray, mask: FaceMask) -> np.ndarray:
    # faceVect = left - right
    # maskVect = mask.left - mask.right
    # faceNorm = np.linalg.norm(faceVect)
    # maskNorm = np.linalg.norm(maskVect)
    # scale = maskNorm / faceNorm
    # faceAngle = np.degrees(np.arctan2(*faceVect))
    # maskAngle = np.degrees(np.arctan2(*maskVect))
    # angle = maskAngle - faceAngle
    # faceCenter = (left+right)/2
    # maskCenter = (mask.left+mask.right) / 2
    # translation = maskCenter - faceCenter
    M = cv2.getAffineTransform(np.array([left, right], np.array([mask.left, mask.right])))
    dst = cv2.warpAffine(face, M, (mask.width, mask.height))
    return dst