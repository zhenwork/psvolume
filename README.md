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
    
## pipeline in submission mode:

1. Load Data:

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
    
2. Image Preprocess


3. Image Merge
    * imageMergeTools.py
    * imageMergeMaster.py 
4. Volume Analysis
    * volumeTools.py
    * volumeProcess.py
    * qshellCorrelation.py
    * projectionViewer.py
