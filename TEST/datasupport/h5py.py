import h5py
import numpy as np 
import utils.filesystem as filesystem


def h5writer(fname, keys, data, chunks=None, opts=7):
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

def h5reader(fname, keys=None):
    """
    Read data from h5 files, default key is the first key.
    The file must exist
    """
    try:
        f = h5py.File(fname, 'r')
        if keys is None: 
            keys = f.keys()[0]
        idata = f[keys].value
        f.close()
        return idata
    except Exception as error:
        print "!! ERROR:", error
        f = None
    
def h5modify(fname, keys, data, chunks=None, opts=7):
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

def h5copy(src=None, dst=None, copy=None, keep=None):
    """
    Goal: data transfer between two h5 files.
    src: source file
    dst: can be an existing and non-existing file
    copy: string or list of strings. copy from src to dst.
    keep: string or list of strings. keep the original data in src.
    """
    if src is None or dst is None: 
        raise Exception, 'error'

    if isinstance(copy, str):
        data = h5reader(src, copy)
        h5modify(dst, copy, data)

    elif isinstance(copy, list) and isinstance(copy[0], str): 
        for i in range(len(copy)):
            data = h5reader(src, copy[i])
            h5modify(dst, copy[i], data)

    elif copy is None:
        if isinstance(keep, str):
            data = h5reader(dst, keep)
            filesystem.copyFile(src=src, dst=dst)
            h5modify(dst, keep, data)
        elif isinstance(keep, list) and isinstance(keep[0], str):
            num = len(keep)
            dataList = []
            for i in range(num):
                dataList.append(h5reader(dst, keep[i]))
            filesystem.copyFile(src=src, dst=dst)
            for i in range(num):
                h5modify(dst, keep[i], dataList[i])
        elif keep is None:
            filesystem.copyFile(src=src, dst=dst)
    else: 
        raise Exception, '!! ERROR'


def h5finder(fname, guessname):
    """
    With a full name of a dataset, just guess one.
    """
    global h5finder_search_name
    h5finder_search_name = guessname
    def tmpGetFullName(cxi_name):
        if cxi_name.endswith(h5finder_search_name):
            return cxi_name
    try:
        fh5finder = h5py.File(fname, 'r')
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

def h5datasets(fname):
    """
    List all dataset names in a h5py file
    """
    global h5datasets_search_name
    h5datasets_search_name = []
    def tmpListDataName(cxi_name):
        h5datasets_search_name.append(cxi_name)
    try:
        fh5list = h5py.File(fname, 'r')
        fh5list.visit(tmpListDataName)
        fh5list.close()
        return h5datasets_search_name
    except Exception as error:
        print "!! ERROR:", error
        fh5list = None


def load(fname):
    """
    load all datasets in a h5py file
    """
    dnames = h5datasets(fname)
    psvmParams = {}
    for each in dnames:
        psvmParams[str(each)] = h5reader(fname, each)
    return psvmParams


def save(fname, params):
    """
    save several datasets to a single h5py file
    """
    for idx, item in enumerate(params):
        if idx == 0:
            h5writer(fname, item, params[item])
        else:
            h5modify(fname, item, params[item])
    return True

def modify(fname, keys, data, chunks=None, opts=7):
    return h5modify(fname, keys, data, chunks=chunks, opts=opts)