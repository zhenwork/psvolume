import os
import importlib  

class dataformat:
    def __init__(FORMAT):
        self.FORMAT = FORMAT
        self.format(self.FORMAT)

    def format(self, FORMAT):
        self.data = None
        self.software_path = os.path.split(os.path.realpath(__file__))[0]
        try:
            self.data = importlib.import_module( "datasupport.%s" % FORMAT )
        except:
            try:
                self.data = importlib.import_module( "psvmusers.datasupport.%s" % FORMAT )
            except:
                raise Exception, "# data format is not supported"
        
        return self

    def load(self, **kwargs):
        return self.data.load(**kwargs)

    def save(self, **kwargs):
        return self.data.save(**kwargs)

    def modify(self, **kwargs):
        return self.data.modify(**kwargs)