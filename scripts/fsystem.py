import h5py
import os,sys
import numpy as np

class H5manager:

    @staticmethod
    def writer(fname, keys=None, data=None, chunks=None, compression=None, compression_opts=7):
        """
        Write data to h5 file. It can be any types of data like string, list, numpy array.
        The function will totally rewrite the file.
        # compression = "gzip"
        """
        try:
            with h5py.File(fname, 'w') as f:
                if keys is not None:
                    if chunks is None:
                        if compression is None:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
                        else:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, compression=compression, compression_opts=compression_opts)
                    else:
                        if compression is None:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks)
                        else:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression=compression, compression_opts=compression_opts)
                    datawr[...] = np.array(data)
        except Exception as error:
            print "!! ERROR:", error 
    
    @staticmethod
    def reader(fname, keys=None):
        """
        Read data from h5 files, default key is the first key.
        The file must exist
        """
        try:
            with h5py.File(fname, 'r') as f:
                if keys is None: 
                    keys = f.keys()[0]
                data = f[keys].value 
            return data
        except Exception as error:
            print "!! ERROR:", error
        
    @staticmethod
    def modify(fname, keys=None, data=None, chunks=None, compression=None, compression_opts=7):
        """
        The data can be an existing or non-existing dataset
        The file must be an existing h5 file.
        """
        try:
            if not os.path.isfile(fname):
                H5manager.writer(fname)
            if keys is not None:
                with h5py.File(fname, 'r+') as f:
                    try: f.__delitem__(keys)
                    except: pass

                    if chunks is None:
                        if compression is None:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype)
                        else:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, compression=compression, compression_opts=compression_opts)
                    else:
                        if compression is None:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks)
                        else:
                            datawr = f.create_dataset(keys, np.array(data).shape, dtype=np.array(data).dtype, chunks=chunks, compression=compression, compression_opts=compression_opts)
                    datawr[...] = np.array(data) 
        except Exception as error:
            print "!! ERROR:", error

    @staticmethod
    def finder(filename, guessname):
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

    @staticmethod
    def dnames(filename):
        """
        List all dataset names in a h5py file
        """
        global h5datasets_search_name
        h5datasets_search_name = []
        def tmpListDataName(cxi_name):
            h5datasets_search_name.append(cxi_name)
        try: 
            fh5list = h5py.File(filename, 'r')
            fh5list.visit(tmpListDataName)
            fh5list.close()
            return h5datasets_search_name
        except Exception as error:
            print "!! ERROR:", error
            fh5list = None

    @staticmethod
    def append(fname, key, data):
        data_tape = H5manager.reader(fname, key)
        if data_tape:
            data_tape = list(data_tape).append(data)
        else:
            data_tape = [data]
        H5manager.writer(fname, key, data_tape)


class PKLmanager:
    @staticmethod
    def writer(fname, params):
        try:
            with open(fname, 'wb') as f:
                pickle.dump(params, f) 
        except Exception as err:
            print "!!! err", err

    @staticmethod
    def reader(fname):
        try: 
            with open(fname, 'rb') as f:
                params = pickle.load(f) 
            return params
        except Exception as err:
            print "!!! err", err

class JSmanager:
    @staticmethod
    def reader(fname):
        try: 
            with open(fname, 'r') as f:
                params = json.load(f)
            return params
        except Exception as err:
            print "!!! err", err

    @staticmethod
    def writer(fname, params, indent=4):
        try: 
            with open(fname, 'w') as f:
                if isinstance(indent,int):
                    json.dump(params, f, indent=indent)
                else:
                    json.dump(params, f)
        except Exception as err:
            print "!!! err", err
            

class PVmanager:
    @staticmethod
    def writer(params, fname):
        for idx, key in enumerate(params):
            if idx == 0:
                H5manager.writer(fname, key, params[key])
                continue
            H5manager.modify(fname, key, params[key])
    
    @staticmethod
    def modify(params, fname):
        for idx, key in enumerate(params):
            if idx == 0:
                H5manager.modify(fname, key, params[key])
                continue
            H5manager.modify(fname, key, params[key])

    @staticmethod
    def reader(fname,keep_keys=None,reject_keys=[]):
        def tempfunc(key, value):
            if "/" in key:
                head = key.strip("/").split("/")[0]
                tail = "/".join(key.strip("/").split("/")[1:]) 
                return {head: temp(tail,value)}
            return {key:value}

        params = {}
        dnames = H5manager.dnames(fname)
        keep_keys = keep_keys or dnames
        keep_keys = set(keep_keys) - set(reject_keys)

        import scripts.utils
        for key in keep_keys:
            scripts.utils.dict_merge(params, tempfunc(key, H5manager.reader(fname, key)))
        return params


class CBFmanager:
    @staticmethod
    def read_data(fname):
        """
        Data is float numpy array 
        """
        content = cbf.read(fname)
        data = np.array(content.data).astype(float)
        return data

    @staticmethod
    def read_header(fname):
        """
        Header is python dict {}
        """
        content = cbf.read(fname, metadata=True, parse_miniheader=True)
        header = content.miniheader
        return header

    @staticmethod
    def reader(fname):
        """
        Data is float numpy array 
        Header is python dict {}
        """
        content = cbf.read(fname, metadata=True, parse_miniheader=True)
        data = np.array(content.data).astype(float)
        header = content.miniheader
        return (data, header)


import re
class Fsystem:

    @staticmethod
    def filename(fname):
        return os.path.basename(fname)

    @staticmethod
    def dirname(fname):
        return os.path.dirname(os.path.abspath(fname))

    @staticmethod
    def filetype(fname):
        """
        strFile = "/a/b", "/a/b.py"
        suffix = "", ".py"
        """
        if not isinstance(fname, str):
            return None 
        if "." not in fname:
            return None
        return fname.split(".")[-1] 

    @staticmethod
    def folder_with_pattern(path=None, pattern="sp*"):
        if path is not None:
            if not os.path.isdir(path):
                return []

        if path is None:
            mpattern = pattern
        else:
            mpattern = os.path.join(path, pattern) 

        select = []
        for name in glob.glob(mpattern):
            if os.path.isdir(name):
                select.append(os.path.abspath(name))
        return sorted(select)

    @staticmethod
    def file_with_pattern(path=None, pattern='*.slice'):
        """
        count the number of files containing *title* in *path*
        """
        if path is not None:
            if not os.path.isdir(path):
                return []

        if path is None:
            mpattern = pattern
        else:
            mpattern = os.path.join(path, pattern) 

        select = []
        for name in glob.glob(mpattern):
            if os.path.isfile(name):
                select.append(os.path.abspath(name))
        return sorted(select)

    @staticmethod
    def copyfile(src=None, dst=None):
        if src is not None:
            from shutil import copyfile
            copyfile(src, dst)