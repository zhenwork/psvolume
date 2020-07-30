import copy
import numpy as np
import dataExtract
import imageTools
import volumeTools
import mergeTools
from numba import jit
from fileManager import H5FileManager
h5M = H5FileManager()
MaskTools   = imageTools.MaskTools()
ScaleTools  = imageTools.ScaleTools()
FilterTools = imageTools.FilterTools()
from scipy.ndimage.filters import median_filter

## data structure
class DataStruct(object):
    def __init__(self):
        self.image = None
        self.mask = 1
        self.peakMask = None
        self.exposureTime = None
        self.phi = None
        self.waveLength = None
        self.pixelSize = None
        self.pixelSizeX = None
        self.pixelSizeY = None
        self.polarization = -1
        self.detectorDistance = None
        self.detectorCenter = None
        self.latticeConstant = None
        self.scale = 1.0
        self.rotAxis = "x"
        self.Amat = None
        self.Bmat = None


## Information of a single diffraction pattern
class ImageAgent(DataStruct):
    def __init__(self, filename=None):
        DataStruct.__init__(self)
        if filename is not None:
            self.loadImage(filename)
        
    def todict(self):
        return self.__dict__
    
    def fromdict(self, entry):
        self.__dict__.update(entry)
    
    def fromobject(self, classobject):
        self = copy.deepcopy(classobject)
        
    def readfile(self, filename=None, fileType=None):
        ## return data in psvm format 
        if filename is None:
            return {}
        psvm = dataExtract.loadfile(filename, fileType=fileType)
        return psvm
    
    def preprocess(self):
        # 1. remove bad pixels
        # self.removeBadPixels(notation=notation)
        # 2. expand mask (True is default) 
        # 3. deep remove bad pixels
        self.deepRemoveBads() 
        # 4. polarization correction
        self.polarizationCorrection()
        # 5. solid angle correction
        self.solidAngleCorrection()
        # 6. remove Bragg peaks
        self.removeBragg()
        return True
    
    
    def loadImage(self, filename, fileType=None):
        psvm = self.readfile(filename, fileType=fileType)
        self.fromdict(psvm)
        return True
    
    def removeBadPixels(self, notation="wtich", vmin=0.001, vmax=100000, rmin=40, rmax=None):
        self.mask *= dataExtract.specialparams(notation=notation)["mask"]
        self.mask *= MaskTools.circleMask(self.image.shape, rmin=rmin, rmax=rmax, center=self.detectorCenter)
        self.mask *= MaskTools.valueLimitMask(self.image, vmin=vmin, vmax=vmax)
        self.image *= self.mask
        self.specMask = self.mask.copy()
        return True
    
    def expandMask(self,expMask=1):
        self.mask = MaskTools.expandMask(self.mask, expandSize=(expMask,expMask), expandValue=0)
        self.image *= self.mask 

    def deepRemoveBads(self):
        newImage, newMask = imageTools.removeExtremes(_image=self.image, algorithm=1, _mask=self.mask, _sigma=15, _window=(11,11))
        self.image = newImage.copy()
        self.mask *= newMask
        return True
    
    def polarizationCorrection(self):
        pscaler = ScaleTools.polarization_scaler(size=self.image.shape, polarization=self.polarization, \
                    detectorDistance=self.detectorDistance, pixelSize=self.pixelSize, center=self.detectorCenter)
        self.image *= pscaler
        # print np.amax(pscaler) = 2.27259170391
        return True
    
    def solidAngleCorrection(self):
        sascaler = ScaleTools.solid_angle_scaler(size=self.image.shape, detectorDistance=self.detectorDistance, \
                                          pixelSize=self.pixelSize, center=self.detectorCenter)
        self.image *= sascaler
        # print np.amax(sascaler) = 6.88204200754
        return True
    
    def scaling(self, reference=None, mode="sum", rmin=160, rmax=400, keepMask=None):
        ## reference is also a Diffraction Object
        if mode == "ave":
            if keepMask is None:
                keepMask = MaskTools.circleMask(self.image.shape, rmin=rmin, rmax=rmax, center=self.detectorCenter)
            sca = np.sum(self.image * self.mask * self.peakMask * keepMask) / np.sum(self.mask * self.peakMask * keepMask)
            ref = np.sum(reference["image"] * reference["mask"] * reference["peakMask"] * keepMask) / np.sum(reference["mask"] * reference["peakMask"] * keepMask)
            self.scale = ref * 1.0 / sca
        elif mode == "sum":
            if keepMask is None:
                keepMask = MaskTools.circleMask(self.image.shape, rmin=rmin, rmax=rmax, center=self.detectorCenter)
            sca = self.image * self.mask * self.peakMask * keepMask
            ref = reference["image"] * reference["mask"] * reference["peakMask"] * keepMask
            self.scale = np.sum(ref) * 1.0 / np.sum(sca)
        elif mode == "rad":
            sca = self.radprofile[rmin:rmax]
            ref = reference["radprofile"][rmin:rmax]
            self.scale = np.dot(ref, sca)/np.dot(sca, sca)
        else:
            raise Exception("!! ERROR IN SCALING")
        return True
    
    def removeBragg(self):
        peakIdenty = mergeTools.PeakMask(Amat=self.Amat, _image=self.image, size=None, xvector=None, \
                                waveLength=self.waveLength, pixelSize=self.pixelSize, center=self.detectorCenter, \
                                detectorDistance=self.detectorDistance, Phi=self.phi, rotAxis=self.rotAxis)
        self.peakMask = 1-peakIdenty
        self.image *= self.peakMask
        return True
    
    def buildPeakMask(self, bmin=0, bmax=0.25):
        peakIdenty = mergeTools.PeakMask(Amat=self.Amat, _image=self.image, size=None, xvector=None, window=(bmin, bmax), \
                                waveLength=self.waveLength, pixelSize=self.pixelSize, center=self.detectorCenter, \
                                detectorDistance=self.detectorDistance, Phi=self.phi, rotAxis=self.rotAxis)
        peakMask = 1-peakIdenty
        # print type(peakMask)
        return peakMask

    def medianBack(self, window=(11,11)):
        return median_filter(self.image, window)

    def circleMask(self, rmin=100, rmax=1000):
        return MaskTools.circleMask(self.image.shape, rmin=rmin, rmax=rmax, center=self.detectorCenter)

    def radprofile(self,window=5):
        aveRadius, aveIntens, sumCount = imageTools.radialProfile(self.image, self.mask * self.peakMask, center=self.detectorCenter, \
                                            vmin=None, vmax=None, rmin=None, rmax=None, stepSize=1, sampling=None, window=window)
        self.radprofile = aveIntens
        return True
    
    def convert2hkl(self):
        hkl = mergeTools.mapImage2Voxel(image=self.image, size=None, Amat=self.Amat, Bmat=None, xvector=None, Phi=self.phi, \
                waveLength=self.waveLength, pixelSize=self.pixelSize, center=self.detectorCenter, detectorDistance=self.detectorDistance, \
                returnFormat="HKL", voxelSize=1.0, rotAxis=self.rotAxis)
        self.hklI = np.zeros((np.prod(self.image.shape),4))
        self.hklI[:,0] = hkl[:,:,0].ravel()
        self.hklI[:,1] = hkl[:,:,1].ravel()
        self.hklI[:,2] = hkl[:,:,2].ravel()
        self.hklI[:,3] = self.image.ravel()
        return True
    
    def mergehkl(self):
        volume, weight = self.merge2volume()
        index = np.where(weight>0)
        volumeScale = volume.copy()
        volumeScale[index] /= weight[index]
        
        self.HKLI = np.zeros((len(index[0]), 4))
        self.HKLIW = np.zeros((len(index[1]), 5))
        
        self.HKLI[:,0] = np.array(index[0]) - 60
        self.HKLI[:,1] = np.array(index[1]) - 60
        self.HKLI[:,2] = np.array(index[2]) - 60
        self.HKLI[:,3] = volumeScale[index]
        
        self.HKLIW[:,0:3] = self.HKLI[:,0:3]
        self.HKLIW[:,3] = volume[index]
        self.HKLIW[:,4] = weight[index]
        return True
    
    def merge2volume(self, keep=False, Vcenter=60, Vsize=121, voxelSize=1.0):
        volume = np.zeros((Vsize,Vsize,Vsize))
        weight = np.zeros((Vsize,Vsize,Vsize))

        volume, weight = mergeTools.Image2Volume(volume=volume, weight=weight, Amat=self.Amat, Bmat=None, _image=self.image, \
                        _mask=self.mask * self.peakMask, keepPeak=False, returnFormat="HKL", xvector=None, waveLength=self.waveLength, \
                        pixelSize=self.pixelSize, center=self.detectorCenter, detectorDistance=self.detectorDistance, \
                        Vcenter=Vcenter, Vsize=Vsize, voxelSize=voxelSize, Phi=self.phi, rotAxis=self.rotAxis)
        
        if keep==True:
            self.volume = volume
            self.weight = weight
            
        return volume, weight

    def convert2reciprocal(self, image=None):
        if image is None:
            return None
        xyz = mergeTools.mapPixel2RealXYZ(size=image.shape, center=self.detectorCenter, pixelSize=self.pixelSize, detectorDistance=self.detectorDistance)
        rep = mergeTools.mapRealXYZ2Reciprocal(xyz=xyz, waveLength=self.waveLength)
        endaxis = len(rep.shape)-1
        return np.sqrt(np.sum(rep**2, axis=endaxis))

    
    def doAction(self, actionName=None, params={}):
        actionObject = getattr(actionCluster, actionName)(params)
        psvm = actionObject.start()
        return psvm
    
    def doActions(self, actions=None):
        for action in actions:
            actionName = action["actionName"]
            params = action["params"]
            status = self.doAction(actionName=actionName, params=params)
            if not status:
                print "!! Error occurs in %s"%actionName
                return False
            print "## Done Action %s"%actionName
        return True
        
        

## A cluster a diffraction patterns
class MergeAgent:
    def __init__(self):
        self.cluster = {}
        self.numImage = 0
        self.startnum = 0
        
    def todict(self):
        return self.__dict__
    
    def fromdict(self, entry):
        self.__dict__.update(entry) 
        
    def readfile(self, filename=None):
        if filename is None:
            return
        psvm = dataExtract.loadfile(filename)
        return psvm
    
    def addfile(self, filename):
        if len(self.cluster) != 0:
            maxIdx = int(max(self.cluster))
        else:
            maxIdx = self.startnum-1
            
        nextIdx = "%.5d"%(maxIdx+1)
        self.cluster[nextIdx] = {"filename":filename}
        self.numImage = len(self.cluster)
        return True
        
    def pushparams(self, params):
        for key in self.cluster:
            self.cluster[key].update(params)
        return True
    
    def merge(self, Vcenter=60, Vsize=121, voxelSize=1.0):
        volume = np.zeros((Vsize,Vsize,Vsize))
        weight = np.zeros((Vsize,Vsize,Vsize))
        
        for each in sorted(self.cluster):
            filename = self.cluster[each]["filename"]
            print "filename: ", filename
            imageAgent = ImageAgent()
            imageAgent.loadImage(filename, fileType="h5")
            imageAgent.image *= imageAgent.scale
            v,w = imageAgent.merge2volume(keep=False, Vcenter=Vcenter, Vsize=Vsize, voxelSize=voxelSize)
            volume += v
            weight += w
            imageAgent = None
            
        self.volume = volume
        self.weight = weight  
        
        return True
    

class Launcher:
    def __init__(self):
        self.a = 1

    def server(self, queue="psanaq", cpus=1, cpus_per_node=1, moniter=True, outDir="./"):
        return 

    def local(self, outDir="./"):
        return 

    def moniter(self):
        return 

    def clean(self):
        return 


class VolumeBase(object):
    def __init__(self, volume=None, weight=None, fname=None):
        self.volume = volume
        self.weight = weight
        self.fname = fname
    def Data(self, dname=None):
        if dname is None:
            return None
        if hasattr(self, dname):
            return getattr(self, dname)
        if self.fname is not None:
            return h5M.h5reader(self.fname, keys=dname)
        else:
            return None

class VolumeAgent(VolumeBase):
    def __init__(self, **kwargs):
        VolumeBase.__init__(self, **kwargs)
    def correlation(self):
        return 
    def symmetrize(self,):
        return
    def dataviewer(self,):
        return
    def radial_background(self,):
        return 