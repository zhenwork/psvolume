import numpy as np
import h5py
import os
from shutil import copyfile

try: import cbf
except: print "!! No cbf package installed."


class H5FileManager:

    def h5writer(self, fname, keys, data, chunks=None, opts=7):
        """
        Write data to h5 file. It can be any types of data like string, list, numpy array.
        The function will totally rewrite the file.
        """
        try:
            f = h5py.File(fname, 'w')
            if chunks is None:
                idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
            else:
                idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression='gzip', compression_opts=opts)
            idatawr[...] = np.array(data)
            f.close()
        except Exception as error:
            print "!! ERROR:", error
            f = None
        
    def h5reader(self, fname, keys=None):
        """
        Read data from h5 files, default key is the first key.
        The file must exist
        """
        try:
            f = h5py.File(fname, 'r')
            if keys is None: keys = f.keys()[0]
            idata = f[keys].value
            f.close()
            return idata
        except Exception as error:
            print "!! ERROR:", error
            f = None
        
    def h5modify(self, fname, keys, data, chunks=None, opts=7):
        """
        The data can be an existing or non-existing dataset
        The file must be an existing h5 file.
        """
        try: 
            f = h5py.File(fname, 'r+')
            try: f.__delitem__(keys)
            except: pass
            
            if chunks is None:
                idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
            else:
                idatawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression='gzip', compression_opts=opts)
            idatawr[...] = np.array(data)
            f.close()
        except Exception as error:
            print "!! ERROR:", error
            f = None

    def h5copy(self, src=None, dst=None, copy=None, keep=None):
        """
        Goal: data transfer between two h5 files.
        src: source file
        dst: can be an existing and non-existing file
        copy: string or list of strings. copy from src to dst.
        keep: string or list of strings. keep the original data in src.
        """
        if src is None or dst is None: 
            raise Exception('error')

        if isinstance(copy, str):
            data = self.h5reader(src, copy)
            self.h5modify(dst, copy, data)

        elif isinstance(copy, list) and isinstance(copy[0], str): 
            for i in range(len(copy)):
                data = self.h5reader(src, copy[i])
                self.h5modify(dst, copy[i], data)

        elif copy is None:
            self.zio = FileSystem()
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

        else: 
            raise Exception('!! ERROR')

    def h5finder(self, filename, guessname):
        """
        With a full name of a dataset, just guess one.
        """
        global h5finder_search_name
        h5finder_search_name = guessname
        def tmpGetFullName(cxi_name):
            if cxi_name.endswith(h5finder_search_name):
                return cxi_name
        try:
            fh5finder = h5py.File(filename, 'r')
            cxiName = fh5finder.visit(tmpGetFullName)
            fh5finder.close()
            if cxiName is None:
                print "## No such dataset"
            else:
                print "## Fit name: ", cxiName
            return cxiName
        except Exception as error:
            print "!! ERROR:", error
            fh5finder = None
            return None

    def h5datasets(self, filename):
        """
        List all dataset names in a h5py file
        """
        def tmpListDataName(cxi_name):
            print cxi_name
        try:
            fh5list = h5py.File(filename, 'r')
            fh5list.visit(tmpListDataName)
            fh5list.close()
        except Exception as error:
            print "!! ERROR:", error
            fh5list = None

    def sliceCut(data, axis='x', window=5, center=None, clim=None):
        """
        input a 3d volume, then it will output the average slice within certain range and angle
        """
        (nx,ny,nz) = data.shape
        if center is None:
            cx = (nx-1.)/2.;
            cy = (ny-1.)/2.;
            cz = (nz-1.)/2.;
        else:
            (cx,cy,cz) = center;
        if clim is None: 
            (vmin, vmax) = (-100, 1000);
        else:
            (vmin, vmax) = clim;
            
        nhalf = (window-1)/2
        Vindex = ((data>=vmin)*(data<=vmax)).astype(float);

        if axis == 'x':
            return np.sum(data[cx-nhalf:cx+nhalf+1,:,:]*Vindex[cx-nhalf:cx+nhalf+1,:,:], axis=0)/(np.sum(Vindex[cx-nhalf:cx+nhalf+1,:,:], axis=0)+1.0e-5)
        elif axis == 'y':
            return np.sum(data[:,cx-nhalf:cx+nhalf+1,:]*Vindex[:,cx-nhalf:cx+nhalf+1,:], axis=1)/(np.sum(Vindex[:,cx-nhalf:cx+nhalf+1,:], axis=1)+1.0e-5)        
        elif axis == 'z':
            return np.sum(data[:,:,cx-nhalf:cx+nhalf+1]*Vindex[:,:,cx-nhalf:cx+nhalf+1], axis=2)/(np.sum(Vindex[:,:,cx-nhalf:cx+nhalf+1], axis=2)+1.0e-5)
        else: 
            return 0

class OtherFileManager:
    def save_pickle(self, params, fileName):
        try:
            f = open(fileName, 'wb')
            pickle.dump(params, f)
            f.close()
            return True
        except:
            return False

    def load_pickle(self, fileName):
        try:
            f = open(fileName, 'rb')
            params = pickle.load(f)
            f.close()
            return params
        except:
            return False

    def load_json(self, fileName):
        try:
            with open(fileName, 'r') as f:
                params = json.load(f)
            return params
        except:
            return False

    def save_json(self, params, fileName):
        try:
            with open(fileName, 'w') as f:
                json.dump(params, f, indent=4)
            return True
        except:
            return False
            

class PsvolumeManager:
    def psvm2h5py(self, psvmParams, fileName):
        h5M = H5FileManager()
        for idx, item in enumerate(psvmParams):
            if idx == 0:
                h5M.h5writer(fileName, item, psvmParams[item])
            else:
                h5M.h5modify(fileName, item, psvmParams[item])
        h5M = None
        return True


class CBFManager:
    def getData(self, fileName):
        """
        Data is float numpy array 
        """
        content = cbf.read(fileName)
        data = np.array(content.data).astype(float)
        return data

    def getHeader(self, fileName):
        """
        Header is python dict {}
        """
        content = cbf.read(fileName, metadata=True, parse_miniheader=True)
        header = content.miniheader
        return header

    def getDataHeader(self, fileName):
        """
        Data is float numpy array 
        Header is python dict {}
        """
        content = cbf.read(fileName, metadata=True, parse_miniheader=True)
        data = np.array(content.data).astype(float)
        header = content.miniheader
        return (data, header)


class FileSystem:

    def fileName(self, strFile):
        return os.path.basename(strFile)

    def baseFolder(self, strFile):
        """
        strFile = "/a/b/c/d", "/a/b.file"
        path = "/a/b/c/", "/a/"
        """
        if not (strFile).endswith('/'): 
            strFile = strFile+'/'
        path = strFile[0:(len(strFile)-strFile[::-1].find('/',1))]
        return path

    def getSuffix(self, strFile):
        """
        strFile = "/a/b", "/a/b.py"
        suffix = "", ".py"
        """
        suffix = strFile[len(strFile)-strFile[::-1].find('.',1)-1:]
        return suffix

    def makeFolderWithPrefix(self, path, title='sp'):
        """
        create folder with starting name *title*
        """
        allFile = os.listdir(path)
        fileNumber = [0]
        for each in allFile:
            if each[:2] == title and each[-4:].isdigit():
                fileNumber.append(int(each[-4:]))
        newNumber = np.amax(fileNumber) + 1
        fnew = os.path.join(path, title+str(newNumber).zfill(4))
        if not os.path.exists(fnew): 
            os.mkdir(fnew)
        return fnew

    def countFilesWith(self, path, title='.slice'):
        """
        count the number of files containing *title* in *path*
        """
        allFile = os.listdir(path)
        counter = 0
        for each in allFile:
            if title in each:
                counter += 1
        return counter

    def listFilesWith(self, path, title='.slice'):
        """
        return a list of files containing *title* in *path*
        """
        allFile = os.listdir(path)
        counter = 0
        selectFile = []
        for each in allFile:
            if title in each:
                counter += 1
                filename = os.path.realpath(path+'/'+each)
                selectFile.append(filename)
        return selectFile.sort()

        # file_name = os.path.realpath(__file__)
        # if (os.path.isfile(file_name)): shutil.copy(file_name, folder_new)

    def copyFile(self, src=None, dst=None):
        if src is not None and dst is not None:
            copyfile(src, dst)