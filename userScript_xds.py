"""
start point of the merging process
 
All unit should be Angstrom (A): wavelength, crystal lattice 
"""

import numpy as np
from imageMergeClient import Lattice2vector

## Tools:
def user_get_xds(filename):
	f = open(filename)
	content = f.readlines()
	f.close()

	Geo = {}
	Geo['pixelSize'] = float(content[7].split()[3])
	Geo['detDistance'] = float(content[8].split()[2])
	Geo['polarization'] = 'y' if float(content[1].split()[3])>0.9 else 'x'
	Geo['wavelength'] = float(content[2].split()[0])
	Geo['center'] = (float(content[8].split()[1]), float(content[8].split()[0]))

	## calculate the invAmat matrix
	invAmat = np.zeros((3,3));
	for i in range(4,7):
	    for j in range(3):
	        invAmat[i-4,j] = float(content[i].split()[j])
	if invAmat is not None:
	    invAmat[1,:] = -invAmat[1,:].copy()
	    tmp = invAmat[:,0].copy()
	    invAmat[:,0] = invAmat[:,1].copy()
	    invAmat[:,1] = tmp.copy()

	## calculate B matrix from lattice constants
	Bmat = np.zeros((3,3));
	a = float(content[3].split()[1]); 
	b = float(content[3].split()[2]); 
	c = float(content[3].split()[3]); 
	alpha = float(content[3].split()[4]); 
	beta  = float(content[3].split()[5]); 
	gamma = float(content[3].split()[6]); 
	(vecx, vecy, vecz, recH, recK, recL) = Lattice2vector(a,b,c,alpha,beta,gamma);
	Bmat = np.array([recH, recK, recL]).T 
	invBmat = np.linalg.inv(Bmat)
	return [Geo, Bmat, invBmat, invAmat]