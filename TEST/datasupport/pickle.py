import pickle
import numpy as np 

def save(params, fname):
    try:
        f = open(fname, 'wb')
        pickle.dump(params, f)
        f.close()
        return True
    except Exception as error:
        print "!! ERROR:", error
        return False

def load(fname):
    try:
        f = open(fname, 'rb')
        params = pickle.load(f)
        f.close()
        return params
    except Exception as error:
        print "!! ERROR:", error
        return False
