"""
os.path.abspath(path)   # get absolute path
os.path.basename(path)  # only keep the file name, remove the path 
os.path.commonprefix(list) # return common *string* from the beginning. For example: "/reg/d/p"
os.path.dirname(path)   # return string before the last "/"
os.path.lexists(path)
os.path.expanduser(path)
os.path.expandvars(path) # Substrings of the form $name or ${name} are replaced by the value of environment variable name
os.path.getatime(path)  # Return the time of last access of path.
os.path.getmtime(path)  # Return the time of last modification of path.
os.path.getctime(path)  # 
os.path.getsize(path)   # Return the size, in bytes, of path.
os.path.isabs(path)     # True if path is an absolute pathname
os.path.isfile(path)
os.path.isdir(path)
os.path.islink(path)
os.path.ismount(path)
os.path.normcase(path)  # 
os.path.normpath(path) 
os.path.realpath(path)  # Similar as abspath(), but realpath() converts hyperlink into real path
os.path.samefile(path1, path2)
os.path.sameopenfile(fp1, fp2)
os.path.samestat(stat1, stat2)
os.path.split(path)   # split file path into two parts: folder, filename
os.path.splitext(fname) # split extension ==> "a/b/file", ".py"


glob.glob("a/b/*.py") # return matched files and folders


re.compile()
re.findall()
re.search()
re.match()
"""


import os
import sys
import glob
from shutil import copyfile

def makeFolderWithPrefix(path, title='sp'):
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


    # file_name = os.path.realpath(__file__)
    # if (os.path.isfile(file_name)): shutil.copy(file_name, folder_new)

def copyFile(src=None, dst=None):
    if src is not None and dst is not None:
        copyfile(src, dst)