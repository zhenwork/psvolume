import os,sys
import time
from fileManager import OtherFileManager, H5FileManager
import subprocess

class Manager:
    def __init__(self, managerfile, labelname):
        self.managerfile = managerfile
        self.labelname = str(labelname)
        self.loadJobs()

    def loadJobs(self):
        ofm = OtherFileManager()
        self.jobs = ofm.load_json(self.managerfile)
        ofm = None
        return 

    def start(self):
        for job_id in self.jobs:
            try:
                job_content = self.jobs[job_id]
                if job_content["type"].lower() == "local":
                    self.local_run(job_id)
                elif job_content["type"].lower() == "server":
                    self.server_run(job_id)
                else:
                    raise Exception("!! No Such Type")
            except:
                print "!!! Not Success: ", job_id 
                break

            print "## %s_%s_%s is running ... "(self.labelname, str(job_id), job_content["job_name"])
            if self.monitor(job_id):
                print "### %s_%s_%s is finished ... "%(self.labelname, str(job_id), job_content["job_name"])
            else:
                print "!!! Not Success: ", job_id 
                break

    def local_run(self, job_id):
        job_content = self.jobs[job_id]
        job
        return 

    def server_run(self, job_id):
        return 

    def monitor(self, job_id):
        job_content = self.jobs[job_id] 
        lock_name = self.labelname + "_"+str(job_id)
        counter = 0
        while True:
            if self.lock_finish(lock_name):
                return True
            elif counter < 10:
                time.sleep(2)
            elif counter < 100:
                time.sleep(5)
            else:
                time.sleep(10)
            counter += 1

    def lock_finish(self, lock_name):
        if os.path.isfile("./.%s"%lock_name):
            return True
        else:
            return False
