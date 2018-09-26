
## basic parameters
Geo = {}
Geo['pixelSize'] = 172.0
Geo['detDistance'] = 200147.4
Geo['polarization'] = 'y'
Geo['wavelength'] = 0.82653
Geo['center'] = (1265.33488372, 1228.00813953)
Geo['Angle_increment'] = 0.1


## inverse U matrix is usually unknown in cctbx result
invAmat = None
invUmat = np.array([[-0.2438,  0.9655,  -0.0919],
					[-0.8608, -0.2591,  -0.4381],
					[-0.4468, -0.0277,   0.8942]])


## B matrix (crystal lattice) in the unit of A-1
Bmat = np.array([[ 0.007369 ,   0.017496 ,   -0.000000],
				 [ 0.000000 ,   0.000000 ,    0.017263],
				 [ 0.015730 ,   0.000000,     0.000000]]).T

sima = Bmat[:,0].copy()
simb = Bmat[:,1].copy()
simc = Bmat[:,2].copy()
lsima = np.sqrt(np.sum(sima**2))
lsimb = np.sqrt(np.sum(simb**2))
lsimc = np.sqrt(np.sum(simc**2))
Kac = np.arccos(np.dot(sima, simc)/lsima/lsimc)
Kbc = np.arccos(np.dot(simb, simc)/lsimb/lsimc)
Kab = np.arccos(np.dot(sima, simb)/lsima/lsimb)
sima = lsima * np.array([np.sin(Kac), 0., np.cos(Kac)])
simb = lsimb * np.array([0., 1., 0.])
simc = lsimc * np.array([0., 0., 1.])
Bmat = np.array([sima,simb,simc]).T
invBmat = np.linalg.inv(Bmat)
