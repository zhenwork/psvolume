# psvolume
1. merge the diffraction pattern into the 3D diffraction volume
2. process the stream file
3. projection view of 3d object


# pipeline
1. Load Data
	* loadDataTools.py
	~ loadData.py
	~ fileManager.py
	~ imagePloter.py
	~ mpidata.py
2. Image Preparation
	* imageProcessTools.py
	~ imageProcessMaster.py
	~ scalingFactor.py
	~ braggPeakCurve.py
	~ removeBraggPeaks.py
	~ removeIsoRing.py
3. Image Merge
	* imageMergeTools.py
	~ imageMergeMaster.py
	~ 
4. Volume Analysis
	* volumeTools.py
	~ volumeProcess.py
	~ qshellCorrelation.py
	~ projectionViewer.py
	~ 

