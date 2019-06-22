import json
import numpy as np

def load(fname):
    try:
        with open(fname, 'r') as f:
            params = json.load(f)
        return params
    except Exception as error:
        print "!! ERROR:", error
        return False

def save(params, fname):
    try:
        with open(fname, 'w') as f:
            json.dump(params, f, indent=4)
        return True
    except Exception as error:
        print "!! ERROR:", error
        return False