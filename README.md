# psvolume

## Purposes:

1. Process diffraction patterns
2. Map diffraction patterns into 3D diffraction volume
3. Process stream files from CrystFEL
4. Projection of 3d objects

## Packages:

    numba (jit)
    cbf
    numpy, scipy
    psana
    
## Pipeline in submission mode:

### 1. Load Data:

Load diffraction patterns and save information to psvolume format. Inputs are file position and indexing results (such as GXPARMS.XDS). The retrived information basically includes:

    diffraction patterns
    detector center
    polarization direction
    Amat ( or astar, bstar, cstar )
    wavelength
    detector distance
    pixel size
    Phi angle

The submission command line is in such format:

    bsub -q queue -x -n 24 -R "span[ptile=12]" -o %J.out mpirun python ./psvolume/scripts-2018-09-25/whereImage.py --xds file-GXPARM.XDS --o ./rawImage --fname ./rawcbf/image_#####.cbf
    
### 2. Image Preprocess
Standard process includes:

    remove detector artifacts
    remove extreme bad pixels
    remove Bragg spots
    polarization correction
    solid-angle correction

The submission command line is in such format:

    bsub -q queue -x -n 12 -R "span[ptile=12]" -o %J.out mpirun python ./psvolume/scripts-2018-09-25/imageProcessMaster.py --i ./rawImage/ --o ./mergeImage
        
### 3. Calculate scaling factor

The general method is to scale each diffraction pattern with water ring intensity.

    bsub -q queue -x -n 12 -R "span[ptile=12]" -o %J.out mpirun python ./psvolume/scripts-2018-09-25/scalingFactor.py --rmin 160 --rmax 400 --o ./scale_0160_0400.h5 --i ./mergeImage --wr 1

### 4. Image Merge

Reject pixels that are mapped in [-0.25, 0.25] around an integer H,K,L. Bad pixels removed/masked/marked in previous steps will also be rejected to the diffraction volume.

    bsub -q queue -x -n 7 -R "span[ptile=7]" -o %J.out mpirun python ./psvolume/scripts-2018-09-25/imageMergeMaster.py --i ./mergeImage/
    
### 5. Volume Analysis

This step basically applys symmetrization, background subtraction to the merged volume, it also converts the raw data to user's defined orientation.

    python ./psvolume/scripts-2018-09-25/volumeProcess.py --i ./sr0001/merge.volume
    
### 6. Resolution dependent correlation

Calculated the correlation of the same dataset or between two datasets.

    python ./psvolume/scripts-2018-09-25/qshellCorrelation.py --i ./sr0001/merge.volume --j ./sr0001/merge.volume --iname anisoData --jname anisoSubRaw --mode shell
