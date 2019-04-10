import os,sys
import time
import shlex
import datetime
import expTools
import imageTools
import mergeTools
import volumeTools
import fileManager
import utils as utils
from numba import jit
MaskTools = imageTools.MaskTools()
ScalingTools = imageTools.ScalingTools()
H5FileManager = fileManager.H5FileManager()

def actionObject(actionName):
    return getattr(sys.modules[__name__], actionName)

class ImageExtraction:
    def __init__(self, params={}):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        inFile = self.params["inFile"]
        fileGXPARMS = self.params["fileGXPARMS"]

        psvmParams = {}
        psvmParams = expTools.cbf2psvm(inFile)
        expInfo = expTools.xdsIndex2psvm(fileGXPARMS)
        psvmParams = utils.mergeDict(old=psvmParams, new=expInfo)

        return psvmParams

class RemoveBadPixels:
    def __init__(self, params={}):
        self.params = params
    def start(self, comm_rank = 0, comm_size=1):
        image = self.params["image"].copy()
        detectorCenter = self.params["detectorCenter"]
        psvmParams = self.params.copy()

        mask = expTools.specificParams()["mask"]
        mask *= MaskTools.circleMask(image.shape, rmin=40, rmax=None, center=detectorCenter)
        mask *= MaskTools.valueLimitMask(image, vmin=0.001, vmax=100000)
        image *= mask

        newImage, newMask = imageTools.removeExtremes(_image=image, algorithm=1, _mask=mask, _sigma=15, _window=(11,11))
        image = newImage.copy()
        mask *= newMask

        psvmParams["image"] = image.copy()
        psvmParams["mask"] = mask.copy()

        return psvmParams
              

class PolarizationCorrection:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        image = psvmParams["image"].copy()
        detectorCenter = psvmParams["detectorCenter"]
        detectorDistance = psvmParams["detectorDistance"]
        pixelSize = psvmParams["pixelSize"]
        polarization = psvmParams["polarization"]

        pscaler = ScalingTools.polarization_scaler(size=image.shape, polarization=polarization, \
                        detectorDistance=detectorDistance, pixelSize=pixelSize, center=detectorCenter)

        image *= pscaler

        psvmParams["image"] = image.copy()
        psvmParams["polarizationScaler"] = pscaler.copy()

        return psvmParams


class SolidAnglePolarizationCorrection:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        image = psvmParams["image"].copy()
        detectorCenter = psvmParams["detectorCenter"]
        detectorDistance = psvmParams["detectorDistance"]
        pixelSize = psvmParams["pixelSize"]  
        polarization = psvmParams["polarization"]

        pscaler = ScalingTools.polarization_scaler(size=image.shape, polarization=polarization, \
                        detectorDistance=detectorDistance, pixelSize=pixelSize, center=detectorCenter)

        sascaler = ScalingTools.solid_angle_scaler(size=image.shape, detectorDistance=detectorDistance, \
                                                  pixelSize=pixelSize, center=detectorCenter)

        scaler = pscaler * sascaler
        scaler /= np.amin(scaler)
        image *= scaler

        psvmParams["polarizationScaler"] = pscaler.copy()
        psvmParams["solidAngleScaler"] = sascaler.copy()
        psvmParams["solidAnglePolarizationScaler"] = scaler.copy()
        psvmParams["image"] = image.copy()

        return psvmParams


class SolidAngleCorrection:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        image = psvmParams["image"].copy()
        detectorCenter = psvmParams["detectorCenter"]
        detectorDistance = psvmParams["detectorDistance"]
        pixelSize = psvmParams["pixelSize"]  

        sascaler = ScalingTools.solid_angle_scaler(size=image.shape, detectorDistance=detectorDistance, \
                                                  pixelSize=pixelSize, center=detectorCenter)

        image *= sascaler

        psvmParams["solidAngleScaler"] = sascaler.copy()
        psvmParams["image"] = image.copy()

        return psvmParams


class RemoveBraggPeaks:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        image = psvmParams["image"].copy()
        mask = psvmParams["mask"].copy()
        detectorCenter = psvmParams["detectorCenter"]
        detectorDistance = psvmParams["detectorDistance"]
        waveLength = psvmParams["waveLength"]
        pixelSize = psvmParams["pixelSize"]
        Phi =  psvmParams["Phi"]
        Amat = psvmParams["Amat"]

        peakMask = mergeTools.PeakMask(Amat=Amat, _image=image, size=None, xvector=None, boxSize=0.25, \
                    waveLength=waveLength, pixelSize=pixelSize, center=detectorCenter, \
                    detectorDistance=detectorDistance, Phi=Phi, rotAxis="x")

        image *= 1-peakMask

        psvmParams["image"] = image.copy()
        psvmParams["mask"] = mask.copy()
        psvmParams["peakMask"] = 1-peakMask

        return psvmParams


class ScalingFactor:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        image = psvmParams["image"].copy()
        mask = psvmParams["mask"].copy()
        detectorCenter = psvmParams["detectorCenter"]

        rmin = psvmParams["ScalingFactor"]["rmin"]
        rmax = psvmParams["ScalingFactor"]["rmax"]

        if "peakMask" in psvmParams:
            peakMask = psvmParams["peakMask"].copy()
            mask *= peakMask

        ROI = MaskTools.circleMask(image.shape, rmin=rmin, rmax=rmax, center=detectorCenter)
        sumIntens = np.sum(image * ROI * mask)
        sumWeight = np.sum(mask * ROI)
        scale = sumIntens/sumWeight 

        psvmParams["scale"] = scale

        return psvmParams

class ImageMerge:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        image = psvmParams["image"].copy()
        mask = psvmParams["mask"].copy()
        detectorCenter = psvmParams["detectorCenter"]
        detectorDistance = psvmParams["detectorDistance"]
        waveLength = psvmParams["waveLength"]
        pixelSize = psvmParams["pixelSize"]
        Phi =  psvmParams["Phi"]
        Amat = psvmParams["Amat"]

        volume = psvmParams["volume"].copy()
        weight = psvmParams["weight"].copy()

        volume, weight = mergeTools.Image2Volume(volume, weight, Amat=Amat, Bmat=None, _image=image, _mask=mask, \
                keepPeak=False, returnFormat="HKL", xvector=None, \
                waveLength=waveLength, pixelSize=pixelSize, center=detectorCenter, detectorDistance=detectorDistance, \
                Vcenter=60, Vsize=121, voxelSize=1., Phi=Phi, rotAxis="x")

        psvmParams["volume"] = volume.copy()
        psvmParams["weight"] = weight.copy()

        return psvmParams


class LaueSymmetrization:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        volume = psvmParams["volume"].copy()

        volumeSym, weightSym = volumeTools.volumeSymmetrize(volume, _threshold=(-100,1000), symmetry="P1211")
        
        psvmParams["volumeSym"] = volumeSym.copy()

        return psvmParams


class FriedelSymmetrization:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        volume = psvmParams["volume"].copy()

        volumeSym, weightSym = volumeTools.volumeSymmetrize(volume, _threshold=(-100,1000), symmetry="friedel")
        
        psvmParams["volumeSym"] = volumeSym.copy()

        return psvmParams        

class RadialBackground:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        volume = psvmParams["volume"].copy()
        Bmat = psvmParams["Bmat"].copy()

        _radialBackground = volumeTools.radialBackground(volume, threshold=(-100,1000), Basis=Bmat)
        volumeSub = volume - _radialBackground
        psvmParams["radialBackground"] = _radialBackground.copy()
        psvmParams["volumeSub"] = volumeSub.copy()

        return psvmParams      

class HKL2volume:
    def __init__(self, params):
        self.params = params
    def start(self, comm_rank=0, comm_size=1):
        psvmParams = self.params.copy()
        volume = psvmParams["volume"].copy()
        Bmat = psvmParams["Bmat"].copy()

        astar = Bmat[:,0].copy()
        bstar = Bmat[:,1].copy()
        cstar = Bmat[:,2].copy()

        volumeXYZ = volumeTools.hkl2volume(volume, astar, bstar, cstar, ithreshold=(-100,1000))
        
        psvmParams["volumeXYZ"] = volumeXYZ.copy()

        return psvmParams           
