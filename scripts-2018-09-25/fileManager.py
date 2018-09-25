import numpy as np
import h5py
import os
from shutil import copyfile

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

	def h5compare(self, src=None, dst=None, copy=None, keep=None):
		if src is None or dst is None: raise Exception('error')

		if isinstance(copy, str):
			data = self.h5reader(src, copy)
			self.h5modify(dst, copy, data)

		elif isinstance(copy, list) and isinstance(copy[0], str): 
			for i in range(len(copy)):
				data = self.h5reader(src, copy[i])
				self.h5modify(dst, copy[i], data)

		elif copy is None:
			self.zio = IOsystem()
			if isinstance(keep, str):
				data = self.h5reader(dst, keep)
				self.zio.copyFile(src=src, dst=dst)
				self.h5modify(dst, keep, data)
			elif isinstance(keep, list) and isinstance(keep[0], str):
				num = len(keep)
				dataList = []
				for i in range(num):
					dataList.append(self.h5reader(dst, keep[i]))
				self.zio.copyFile(src=src, dst=dst)
				for i in range(num):
					self.h5modify(dst, keep[i], dataList[i])
			elif keep is None:
				self.zio.copyFile(src=src, dst=dst)

		else: raise Exception('error')

	def readtxt(self, path):
		f = open(path)
		content = f.readlines()
		f.close()
		return content

class IOsystem:
	def get_path_folder(self, strFile):
		if not (strFile).endswith('/'): strFile = strFile+'/'
		path = strFile[0:(len(strFile)-strFile[::-1].find('/',1))]
		folder = strFile
		return [path, folder]

	def get_suffix(self, strFile):
		suffix = strFile[len(strFile)-strFile[::-1].find('.',1)-1:]
		return suffix

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
		selectFile = []
		for each in allFile:
			if title in each:
				counter += 1
				selectFile.append(path+'/'+each)
		return [counter, selectFile]

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

	def copyFile(self, src=None, dst=None):
		if src is not None and dst is not None:
			copyfile(src, dst)