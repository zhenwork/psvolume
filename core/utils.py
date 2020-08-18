import collections

def dict_merge(x1, x2):
    for key, val in x2.iteritems():
        if isinstance(val, collections.Mapping):
            x1[key] = merge(x1.get(key, {}), val)
        else:
            x1[key] = val
    return x1

def dotproduct(x, y):
    return sum((a*b) for a, b in zip(x, y))
    
def length(arr):
    return np.linalg.norm(arr)

def angle(x,y):
    return np.arccos(x.dot(y)/length(x)/length(y)) * 180. / np.pi

