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

def mergeDict(old=None, new=None):
    if new is None:
        return old

    for item in new:
        old[item] = new[item]

    return old
