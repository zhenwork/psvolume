"""
1. utils functions
"""
import time
import shlex
import datetime
import subprocess

def getTime():
    """ 
    return accurate time point in format: Year-Month-Day-Hour:Minute:Second.unique_labels
    """ 
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S.%f')

def getUserName(): 
    cmd = "whoami" 
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return out


def getevents(runs):
    rlist = []
    if not isinstance(runs,str):
        return rlist
    for each in runs.split(","):
        if each == "":
            continue
        elif ":" in each:
            start,end = each.split("-")
            rlist.extend(range(int(start),int(end)+1))
        else:
            rlist.append(int(each))
    return rlist
