import numpy as np
import h5py
import os

class iFile:
	def h5writer(self, fname, keys, data, chunks=None, opts=7):
		f = h5py.File(fname, 'w')
		if chunks is None:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
		else:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression='gzip', compression_opts=opts)
		idatawr[...] = np.array(data)
		f.close()
		
	def h5reader(self, fname, keys=None):		
		f = h5py.File(fname, 'r')
		if keys is None: keys = f.keys()[0]
		idata = f[keys].value
		f.close()
		return idata
		
	def h5modify(self, fname, keys, data, chunks=None, opts=7):
		f = h5py.File(fname, 'r+')
		try: f.__delitem__(keys)
		except: pass
		
		if chunks is None:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
		else:
			idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression='gzip', compression_opts=opts)
		idatawr[...] = np.array(data)
		f.close()

	def makeFolder(self, path, title='sp'):
		allFile = os.listdir(path)
		fileNumber = [0]
		for each in allFile:
			if each[:2] == title and each[-4:].isdigit():
				fileNumber.append(int(each[-4:]))
		newNumber = np.amax(fileNumber) + 1
		fnew = os.path.join(path, title+str(newNumber).zfill(4))
		if not os.path.exists(fnew): os.mkdir(fnew)
		return fnew

	def counterFile(self, path, title='.slice'):
		allFile = os.listdir(path)
		counter = 0
		for each in allFile:
			if title in each:
				counter += 1
		return counter

		# file_name = os.path.realpath(__file__)
		# if (os.path.isfile(file_name)): shutil.copy(file_name, folder_new)

	def get_image_info(self, path):
		f = h5py.File(path, 'r')
		Info = {}
		Info['readout'] = f['readout'].value
		Info['waveLength'] = f['waveLength'].value
		Info['polarization'] = f['polarization'].value
		Info['detDistance'] = f['detDistance'].value
		Info['pixelSize'] = f['pixelSize'].value
		Info['center'] = f['center'].value
		Info['exp'] = f['exp'].value
		Info['run'] = f['run'].value
		Info['event'] = f['event'].value
		Info['rotation'] = f['rotation'].value
		Info['rot'] = f['rot'].value
		Info['scale'] = f['scale'].value
		Info['Smat'] = f['Smat'].value
		f.close()
		return Info

	def readtxt(self, path):
		f = open(path)
		content = f.readlines()
		f.close()
		return content