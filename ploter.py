import numpy as np
from PyQt4 import QtGui
import pyqtgraph as pg
import os, sys
import matplotlib.pyplot as plt


class iPloter:
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