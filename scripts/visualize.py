import numpy as np
from PyQt4 import QtGui
import pyqtgraph as pg
import sys
import matplotlib.pyplot as plt


class PlotTools:
    def pgplot(self, idata, clim=None):
        dataff = idata.copy()
        if clim is None:
            print('automatic parameter finding ... ')
            datastring = dataff.copy()
            datastring.shape = np.prod(dataff.shape),
            datastring = np.sort(datastring)
            length = len(datastring)
            mind = length*0.98-1
            mind = int(mind)
            maxmax = datastring[mind]
            minmin = 0
            if maxmax <= minmin:
                minmin = np.amin(datastring)

        else:
            minmin = clim[0]
            maxmax = clim[1]

        pg.image(dataff, title="crystal rotation series",levels=(minmin, maxmax))
        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            pg.QtGui.QApplication.exec_()

    def imshow(self, image, clim=None, size=None, subPlot=None):
        if not isinstance(image, list)
            plt.figure(figsize=size)
            plt.imshow(image, clim=clim)
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=size)
            M,N = subPlot
            idx = 1
            for i in range(M):
                for j in range(N):
                    plt.subplot(M,N,idx)
                    plt.imshow(image[idx-1], clim=clim)
                    plt.tight_layout()
                    idx += 1
            plt.tight_layout()
            plt.show()


    def pltRing(self, arr, r = None, unit=None, center=(None,None), spacing=None, clim=None):
        (nx, ny) = arr.shape
        cx = center[0];
        cy = center[1];
        if cx is None: cx = (nx-1.)/2.
        if cy is None: cy = (ny-1.)/2.
        if spacing is None:
            spacing = max(abs(nx-cx), abs(ny-cy), abs(cx), abs(cy))/5.
            numRing = 5
        else:
            numRing = int( max(abs(nx-cx), abs(ny-cy), abs(cx), abs(cy))/spacing)

        if r is not None and unit is not None:
            r = r*1.0/unit

        if r is None:
            rad = np.linspace(0,2*np.pi, 1000)
            for i in range(numRing):
                xaxis = spacing*(i+1)*np.sin(rad)+cx
                yaxis = spacing*(i+1)*np.cos(rad)+cy
                if i==0:
                    x = xaxis.copy();
                    y = yaxis.copy();
                else:
                    x = np.append(x, xaxis);
                    y = np.append(y, yaxis);                    
        else:
            rad = np.linspace(0,2*np.pi, 1000)
            spacing = r
            x = spacing*np.sin(rad)+cx
            y = spacing*np.cos(rad)+cy


        plt.figure(figsize=(8,8));
        plt.imshow(arr, clim=clim)
        plt.plot(x, y, '--', linewidth=2,color='black')
        plt.title('spacing r='+str(round(spacing, 2)))
        plt.tight_layout()
        plt.show()

