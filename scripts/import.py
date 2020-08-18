# python --from image=file1.cbf,file2.cbf,file3.cbf dials=file4 --to diffuse.data

import os,sys
import core.filesystem
import diffuse.datafile

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-from","--from", help="cbf=file1,file2 json=file3",default=[],nargs="*",type=str) 
parser.add_argument("-name","--name", help="data/image=file1/data/image",default=[],nargs="*",type=str) 
parser.add_argument("-into","--into", help="diffuse.data",default="./diffuse.data",type=str) 
args = parser.parse_args() 

args.from = args.from.split() 
args.name = args.name.split() 

## check whether we have files
if len(args.from)==0 and len(args.name)==0:
    print "!! No input files ... "
    sys.exit(1)


## load experiment files
core.filesystem.H5manager.writer(args.into,keys=None,data=None)
for file in args.from:
    ftype,fnames = file.split("=")[0]
    fnames = fnames.split(",")
    if ftype.lower() == "imagefile":
        core.filesystem.H5manager.modify(args.into,keys="data/imagefile",data=fnames)
        print("## importing: %d imagefiles"%len(fnames))
        continue
    if ftype.lower() == "backgfile":
        core.filesystem.H5manager.modify(args.into,keys="data/backgfile",data=fnames)
        print("## importing: %d backgfiles"%len(fnames))
        continue
    for fname in fnames:
        if ftype.lower() in ["json","gxparms","dials_expt","dials_report"]:
            data = diffuse.datafile.loadfile(fname, fileType=ftype.lower()) 
            core.filesystem.PVmanager.modify(params={"data":data},fname=args.into)
            print("## importing: %s"%len(fname))
        else:
            print("!! doesn't support %s for %s"%(ftype,fname))


## load data into data names
for file in args.name:
    data_name,reader = file.split("=")
    if "/" in reader:
        read_file = reader.split("/")[0]
        read_name = "/".join(reader.split("/")[1:])
        core.filesystem.H5manager.modify(args.into,keys=data_name, \
                data=core.filesystem.H5manager.reader(read_file,read_name) )
        print("## importing: %s"%len(reader))
    else:
        # load all data inside the read_file
        read_file = reader
        for read_name in core.filesystem.H5manager.dnames(read_file):
            core.filesystem.H5manager.modify(args.into,keys="%s/%s"(data_name,read_name), \
                    data=core.filesystem.H5manager.reader(read_file,read_name) )
        print("## importing: %s"%len(reader))


# save history into args.into
from sys import argv
command = "python "+" ".join(argv)
core.filesystem.H5manager.append(args.into,keys="history",data=command)
