import time
import os,sys
import random
import string
import datetime
import subprocess


def random_string(N=20):
    return ''.join([random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N)])

def date_string():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S.%f')

def getusername(): 
    cmd = "whoami" 
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return out

