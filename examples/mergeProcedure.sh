1. Input userScript.py

2. Run whereImage.py
	bsub -q psfehq -x -n 32 -R "span[ptile=16]" -o %J.out mpirun python ~/software/psvolume/whereImage.py --xds ./XPARM.XDS-ZHEN --o ./rawImage --fname /reg/data/ana04/users/zhensu/xpptut/volume/ICHg150t2/crystal/rawcbf/I4ichg150t2_13_#####.cbf
	
	This process will save images to h5py format and also save users information to each image in rawImage folder

3. Run imageProcessMaster.py
	bsub -q psfehq -x -n 100 -R "span[ptile=10]" -o %J.out mpirun python ~/software/psvolume/imageProcessMaster.py --i ./rawImage/ --o ./mergeImage/

	This script preprocesses raw images and save them to mergeImage folder, this part is slow

4. Run imageMergeMaster.py
	bsub -q psfehq -x -n 7 -R "span[ptile=7]" -o %J.out mpirun python ~/software/psvolume/imageMergeMaster.py --i ./mergeImage/
	This is the main script for image merging. It is actually not very slow.

5. Run volumeProcess.py
	python ~/software/psvolume/volumeProcess.py --i ./sr0001/merge.volume

6. Run qshellCorrelation.py
	python ~/software/psvolume/qshellCorrelation.py --i ./sr0001/merge.volume -j /reg/data/ana04/users/zhensu/xpptut/experiment/0023/0018/volumelist_121.h5 --rmax 50 --mode ball

If you donnot have scalingFactor, use scale factor as 1 and then run 1-2-3 -7(wr=True) -3-4-5-6
7. Run scalingFactor.py
	bsub -q psfehq -x -n 40 -R "span[ptile=8]" -o %J.out mpirun python ~/software/psvolume/scalingFactor.py --rmin 0 --rmax 160 --o ./scale_0000_0160.h5 --i ./mergeImage/

Sometimes you need to modify the image files:
8. Run imageModify.py
	bsub -q psfehq -x -n 24 -R "span[ptile=8]" -o %J.out mpirun python psvolume/imageModify.py --o ./mergeImage