import json
import time
import os,sys
import pickle
import random
import inspect
import datetime
import numpy as np

print("import dataobject:",datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

class FileHandler:
    def __init__(self,ftype):
        self.ftype = ftype
    def get(self,fname=None,dname=None,layer=None):
        if self.ftype in ["numpy","npy","np"]:
            return np.load(fname)
        else:
            return pickle.load(open(fname,"rb"))
    def save(self,fname=None,data=None):
        pickle.dump(data,open(fname,"wb")) 
        
class Fdata:
    """
    1. data=Fdata(); data[1:]=array; data()
    2. data save into pickle
    3. support operations: +, -, *, /, **, +=, -=, *=, /=, +, -, <, <=, !=, ==, >, >=
    """
    def __init__(self,__data__=None,__fname__=None,__ftype__=None,__dname__=None,__layer__=None):
        self.__data__  = __data__
        self.__fname__ = __fname__
        self.__ftype__ = __ftype__
        self.__dname__ = __dname__
        self.__layer__ = __layer__
    def __getitem__(self,item):
        if self.__data__ is None:
            tmpData = self.reader(item)
        elif hasattr(self.__data__, "__getitem__"):
            tmpData = self.__data__[item]
        else:
            tmpData = self.__data__
        return Fdata(tmpData)
    def __setitem__(self,item,other):
        tmpData = np.array(self.__call__()) 
        if type(other) is Fdata:
            tmpData[item] = np.array(other())
        else:
            tmpData[item] = np.array(other)
        if isinstance(self.__data__, (list,tuple)):
            self.__data__ = type(self.__data__)(tmpData)
        else:
            self.__data__ = tmpData
        self.cleanfile()
    def __call__(self):
        if self.__data__ is None:
            return self.reader()
        return self.__data__
    def reader(self,item=None):
        if self.__fname__ is None:
            return None
        if self.__layer__ is None:
            return FileHandler(self.__ftype__).get(fname=self.__fname__,dname=self.__dname__,layer=item)
        if item is not None:
            tmpData = FileHandler(self.__ftype__).get(fname=self.__fname__,dname=self.__dname__,layer=self.__layer__)
            if hasattr(tmpData,"__getitem__"):
                return tmpData[item]
            return tmpData
        return FileHandler(self.__ftype__).get(fname=self.__fname__,dname=self.__dname__,layer=self.__layer__)
    def cleanfile(self):
        self.__fname__ = None
        self.__ftype__ = None
        self.__dname__ = None
        self.__layer__ = None
        return self
    def release(self,fname=None,folder=None,header=None):
        """
        1. if __fname__ is not None, then it mean the data has never been changed
        2. if __data__ is None, then skip things, because the data position keeps the same
        """
        if self.__fname__ is not None:
            self.__fname__ = os.path.realpath(self.__fname__)
        elif self.__data__ is None:
            pass
        elif fname is not None:
            FileHandler("pickle").save(fname=os.path.realpath(fname),data=self.__data__)
            self.__init__(None,os.path.realpath(fname),"pickle",None,None)
        elif folder is not None:
            while True:
                fname = os.path.join(folder,(header or "temp")+"_%.6d"%random.randint(0,1e6)+".fdata")
                if not os.path.isfile(fname):
                    break
            self.release(fname,None,None)
        else:
            os.path.isdir("./tempDir") or os.makedirs("./tempDir")
            self.release(None,"./tempDir",header)
        self.__data__ = None
        return self
    # operation: +,-,*,/,**
    def __add__(self,other):
        if type(other) is Fdata:
            return self.__add__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)):
            return Fdata(type(tmpData)(np.array(tmpData) + np.array(other)))
        elif isinstance(other,(list,tuple)):
            return Fdata(type(other)(np.array(tmpData) + np.array(other)))
        return Fdata(tmpData + other)
    def __sub__(self,other):
        if type(other) is Fdata: 
            return self.__sub__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)):
            return Fdata(type(tmpData)(np.array(tmpData) - np.array(other)))
        elif isinstance(other,(list,tuple)):
            return Fdata(type(other)(np.array(tmpData) - np.array(other)))
        return Fdata(tmpData - other)
    def __mul__(self,other):
        if type(other) is Fdata: 
            return self.__mul__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)):
            return Fdata(type(tmpData)(np.array(tmpData) * np.array(other)))
        elif isinstance(other,(list,tuple)):
            return Fdata(type(other)(np.array(tmpData) * np.array(other)))
        return Fdata(tmpData * other)
    def __truediv__(self,other): 
        if type(other) is Fdata: 
            return self.__truediv__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)):
            return Fdata(type(tmpData)(np.array(tmpData) / np.array(other)))
        elif isinstance(other,(list,tuple)):
            return Fdata(type(other)(np.array(tmpData) / np.array(other)))
        return Fdata(tmpData / other)
    def __pow__(self,other): 
        if type(other) is Fdata: 
            return self.__pow__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)):
            return Fdata(type(tmpData)(np.array(tmpData) ** np.array(other)))
        elif isinstance(other,(list,tuple)):
            return Fdata(type(other)(np.array(tmpData) ** np.array(other)))
        return Fdata(tmpData ** other)
    
    ## operation: +=, -=, *=, /=
    def __iadd__(self, other):
        self.__data__ = (self + other)() 
        return self.cleanfile()
    def __isub__(self, other):
        self.__data__ = (self - other)()
        return self.cleanfile()
    def __imul__(self, other):
        self.__data__ = (self * other)()
        return self.cleanfile()
    def __idiv__(self, other):
        self.__data__ = (self / other)()
        return self.cleanfile()
        
    ## operation: (-), (+), abs, int, float. 
    def __neg__(self):
        return Fdata(0) - self
    def __pos__(self):
        return self
    def __abs__(self):
        tmpData = self.__call__()
        if isinstance(tmpData, (list,tuple)):
            return Fdata(type(tmpData)(list(np.abs(np.array(tmpData)))))
        return Fdata(abs(tmpData))
    def __int__(self):
        tmpData = self.__data__ or self.reader()
        if isinstance(tmpData, (list,tuple)):
            return Fdata(type(tmpData)(list(np.array(tmpData).astype(int))))
        return Fdata(int(tmpData))
    def __float__(self):
        tmpData = self.__data__ or self.reader()
        if isinstance(tmpData, (list,tuple)):
            return Fdata(type(tmpData)(list(np.array(tmpData).astype(float))))
        return Fdata(float(tmpData))
    
    # operation: <, <=, ==, !=, >=, >
    def __lt__(self, other):
        if type(other) is Fdata:
            return self.__lt__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)) or isinstance(other,(list,tuple)):
            return type(tmpData)(np.array(tmpData) < np.array(other))
        else:
            return tmpData < other
    def __le__(self, other):
        if type(other) is Fdata:
            return self.__le__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)) or isinstance(other,(list,tuple)):
            return type(tmpData)(np.array(tmpData) <= np.array(other))
        else:
            return tmpData <= other
    def __eq__(self, other):
        if type(other) is Fdata:
            return self.__eq__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)) or isinstance(other,(list,tuple)):
            return type(tmpData)(np.array(tmpData) == np.array(other))
        else:
            return tmpData == other
    def __ne__(self, other):
        if type(other) is Fdata:
            return self.__ne__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)) or isinstance(other,(list,tuple)):
            return type(tmpData)(np.array(tmpData) != np.array(other))
        else:
            return tmpData != other
    def __ge__(self, other):
        if type(other) is Fdata:
            return self.__ge__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)) or isinstance(other,(list,tuple)):
            return type(tmpData)(np.array(tmpData) >= np.array(other))
        else:
            return tmpData >= other
    def __gt__(self, other):
        if type(other) is Fdata:
            return self.__gt__(other())
        tmpData = self.__call__()
        if isinstance(tmpData,(list,tuple)) or isinstance(other,(list,tuple)):
            return type(tmpData)(np.array(tmpData) > np.array(other))
        else:
            return tmpData > other
        

class Tree:
    def __new__(cls,data=None):
        if isinstance(data,dict):
            if "__fname__" in data:
                return Fdata(**data)
            return super(Tree,cls).__new__(cls)
        elif isinstance(data,(list,tuple,set)):
            return type(data)([Tree(x) for x in data])
        else:
            return data
    def __init__(self,data=None):
        for key in data:
            setattr(self, key, Tree(data[key]))
    def __detach__(self):
        return True
    def __getitem__(self,item):
        """
        1. Always consider Fdata as one object, doesn't consider Fdata[item], for example, all images use the same Fdata(mask)
            if Fdata is also considered, then you will return a half mask
        2. detachable: list of uniform objects, Tree() with True __detach__, other objects with __detach__
        3. image:[1,2,3] considered as 1, while image:[[1,2,3]] considered as any item slice, please define the correct structure ... 
        """
        out = Tree({})
        keys = self.__dict__.keys() 
        for key in keys:
            data = getattr(self,key)
            if isinstance(data, (list,tuple)) and len(data)>0:
                if len(data) == 1:
                    if isinstance(item, int):
                        setattr(out, key, data[0])
                    elif hasattr(item,"__len__") and len(item) == 1:
                        setattr(out, key, data[0])
                    else:
                        setattr(out, key, data)
                else:
                    setattr(out, key, data[item])
            elif hasattr(data,"__detach__") and data.__detach__() and hasattr(data,"__getitem__"):
                setattr(out, key, data[item])
            else:
                setattr(out, key, data)
        return out
    def __setitem__(self,item,value):
        """
        1. value is a number, then set the whole branch to number
        2. value is a dict/object, then assign every key
        3. some keys are missing
        4. change an item, so expand the list
        """
        branch = self.__getitem__(0)
        maxLength = self.maxlength()
        if isinstance(value, dict):
            for key,data in value.items():
                if isinstance(data,dict):
                    return 
                elif isinstance(data,(int,float,str,bool,set,tuple,list)):
                    setattr(self,key,data)
    def todict(self):
        """
        1. convert class object into dictionary, but keep other data types, such as numpy array
        """
        def data2dict(data=None):
            if data is None:
                return None
            elif isinstance(data, (str, bool, int, float)):
                return data
            elif type(data) is Fdata:
                return dict((x,y) for x,y in data.release().__dict__.items() if y is not None)
            elif hasattr(data, "__dict__"):
                return data2dict(data.__dict__)
            elif isinstance(data, (set, tuple, list)):
                return type(data)([data2dict(x) for x in data])
            elif isinstance(data, dict):
                return dict((x,data2dict(y)) for x,y in data.items())
            return data
        return data2dict(self.__dict__.copy())
    def release(self,folder=None,header=None):
        """
        1. save all data types that are not serializable.
        2. you can do data.release().todict(), and save this into json
        """
        def datarelease(data=None):
            if data is None:
                return None
            elif isinstance(data, (bool, int, float)):
                return data
            elif type(data) is Fdata:
                return data.release(folder=folder, header=header)
            elif isinstance(data, (set, list, tuple)):
                return type(data)([datarelease(x) for x in data])
            elif isinstance(data, dict):
                return dict((x,datarelease(y)) for x,y in data.items())
            elif hasattr(data,"__dict__"):
                for key,value in data.__dict__.items():
                    setattr(data, key, datarelease(value))
                return data
            else:
                return Fdata(data).release(folder=folder, header=header)
        return datarelease(self)
    # operation: len
    def maxlength(self,data=None):
        maxLength = 0
        fakLength = []
        if isinstance(data, (list,tuple)):
            if len(data) == 0:
                maxLength = max(maxLength,0)
            elif isinstance(data[0],(int, float, str, bool, list, type(None))):
                datatype = type(data[0])
                if all(isinstance(x,datatype) for x in data):
                    fakLength.append(len(data))
                    maxLength = max(maxLength, 1)
                else:
                    maxLength = max(maxLength, len(data))
            else:
                maxLength = max(maxLength, len(data))
        elif isinstance(data, dict):
            for x,y in data.items():
                maxl,fakl = self.maxlength(y)
                maxLength = max(maxl,maxLength)
                fakLength.extend(fakl)
        elif hasattr(data,"__detach__") and data.__detach__():
            maxl,fakl = self.maxlength(data.__dict__)
            maxLength = max(maxl,maxLength)
            fakLength.extend(fakl)
        else:
            maxLength = max(maxLength,1)
        return maxLength,fakLength
    def __len__(self):
        maxLength,fakLength = self.maxlength(self) 
        if maxLength > 1:
            return maxLength
        elif len(fakLength) == 0:
            return maxLength
        elif max(fakLength) <= maxLength:
            return maxLength
        elif maxLength == 0:
            raise NotImplemented
        else:
            return max(fakLength)
#     # operation: +,-,*,/,**
#     def __add__(self,other): 
#     def __sub__(self,other): 
#     def __mul__(self,other): 
#     def __truediv__(self,other):  
#     def __pow__(self,other):  
    
#     ## operation: +=, -=, *=, /=
#     def __iadd__(self, other): 
#     def __isub__(self, other): 
#     def __imul__(self, other): 
#     def __idiv__(self, other): 
        
#     ## operation: (-), (+), abs, int, float. 
#     def __neg__(self): 
#     def __pos__(self): 
#     def __abs__(self): 
#     def __int__(self): 
#     def __float__(self): 
    
#     # operation: <, <=, ==, !=, >=, >
#     def __lt__(self, other): 
#     def __le__(self, other): 
#     def __eq__(self, other): 
#     def __ne__(self, other): 
#     def __ge__(self, other): 
#     def __gt__(self, other): 
    
    