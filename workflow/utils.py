import os,sys

PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)
    
import core.fsystem 
import diffuse.datafile 

class 
def argument_list_from_file(from_file_input):
    # ["file:x>y,z>d", "file2"]
    