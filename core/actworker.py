import json
import os,sys 
import numpy as np


class WorkerFunction:
    # Run actobj.start() -> returns
    # basic: start(),status(),ready(),success(),wait(),returns(),clear()
    # status: waiting,running,done,failed,terminated
    def __init__(self,actpack=None):
        self.actpack = actpack
        self.starttime = time.time()

    def start(self):
        assert self.actpack.get("actobj")
        actobj = self.actimport()
        from multiprocessing.pool import ThreadPool
        self.pool = ThreadPool(processes=1)
        if hasattr(actobj,"start"):
            self.result = self.pool.apply_async(actobj(self.actpack.get("params")).start, ())
        else:
            self.result = self.pool.apply_async(actobj(self.actpack.get("params")).run, ())
        self.pool.close()

    def actimport(self):
        # check obj/dict(name,pickle)
        actobj = self.actpack.get("actobj")
        if not isinstance(actobj,dict):
            return actobj
        if actobj.get("name"):
            import imp
            obj = imp.load_source('', actobj.get("path"))
            return getattr(obj,actobj.get("name"))
        if actobj.get("pickle"):
            return pickle.load(open(actobj.get("pickle"),"rb"))
        raise Exception("!! Not valid for WorkerFunction")

    def status(self):
        if not hasattr(self,"result"):
            return "waiting"
        if self.result.ready():
            if self.result.successful():
                return "done"
            return "failed"
        return "running"

    def ready(self):
        # waiting,running,pending,suspended,failed,terminated,done
        if self.status() not in ["running","waiting","pending","suspended"]:
            return True
        return False

    def success(self):
        if self.status() in ["done"]:
            return True
        return False

    def returned(self):
        return_data = self.result.get()
        self.pool.close()
        self.pool.terminate()
        if isinstance(return_data,dict):
            return return_data
        return {"return":return_data}

    def close(self):
        try: self.pool.close()
        except: pass
        try: self.pool.terminate()
        except: pass
        try: self.pool.join()
        except: pass

    def wait(self):
        try: self.result.get()
        except: pass

    def clear(self):
        self.close()
        for key in self.__dict__:
            setattr(self,key,None)

    def runtime(self):
        return time.time() - self.starttime


class WorkerCommand:
    def __init__(self,actpack=None):
        self.actpack = actpack
        self.starttime = time.time()
        self.submited = False
        self.submitus = False
        self.submitout = None
        self.submiterr = None

    def start(self):
        command = self.makecmd()
        print "command", command
        try:
            import shlex
            self.submited = True
            self.p = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.submitus = True
        except Exception as err: 
            self.submiterr = err

    def _cpus(self):
        return self.actpack.get("cpus") or 1

    def status(self):
        if not self.submited:
            return "waiting"
        if not self.submitus:
            return "failed"
        elif self.p.poll() is None:
            return "running"
        elif self.p.poll() > 0:
            return "failed"
        elif self.p.poll() < 0:
            return "terminated"
        # self.p.poll() == 0: return "done"
        if not self.actpack.get("freturn"):
            return "done"
        if not os.path.isfile(self.actpack.get("freturn")):
            return "failed"
        return "done"

    def makecmd(self):
        actobj = self.actpack.get("actobj")
        if actobj.get("command"):
            return actobj.get("command")
        elif actobj.get("flaunch") and self._cpus()>1:
            command = "mpirun -n %d python %s"%(self._cpus(),actobj.get("flaunch"))
            return command
        elif actobj.get("flaunch") and self._cpus()==1:
            command = "python %s"%actobj.get("flaunch")
            return command
        raise NotImplemented

    def ready(self):
        # waiting,running,pending,suspended,failed,terminated,done
        if self.status() not in ["running","waiting","pending","suspended"]:
            return True
        return False

    def success(self):
        if self.status() in ["done"]:
            return True
        return False

    def returned(self):
        if self.status() in [None,"running","waiting","pending","suspended"]:
            return {"return":None}
        elif self.status() in ["failed","terminate"]:
            out,err = None,None
            try: out,err = self.p.communicate()
            except: pass
            self.submitout = self.submitout or out
            self.submiterr = self.submiterr or err
            return {"out":self.submitout,"err":self.submiterr}
        elif self.status() in ["done"]:
            out,err = None,None
            try: out,err = self.p.communicate()
            except: pass
            self.submitout = self.submitout or out
            self.submiterr = self.submiterr or err
            return {"out":self.submitout,"err":self.submiterr}
        raise NotImplemented
    
    def jobreturn(self):
        if self.actpack.get("freturn") and self.status()=="done":
            import pickle
            return pickle.load(open(self.actpack.get("freturn"),"rb"))
        return None

    def wait(self):
        try: 
            out, err = self.p.communicate()
            self.submitout = out
            self.submiterr = err
        except: pass

    def close(self):
        try: self.p.terminate()
        except: pass
        try: self.p.kill()
        except: pass
        try: self.p.wait()
        except: pass

    def clear(self):
        self.close()
        for key in self.__dict__:
            setattr(self,key,None)

    def runtime(self):
        return time.time() - self.starttime


class WorkerBsubPsana:
    def __init__(self,actpack=None):
        self.actpack = actpack
        self.starttime = time.time() 
        self.submited = False
        self.submitus = False 
        self.submitout = None 
        self.submiterr = None 
        self.__status__ = "waiting"

    def start(self):
        command = self.makecmd()
        try:
            import shlex
            self.submited = True
            self.p = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.submitus = True
        except Exception as error:
            self.submiterr = error

    @staticmethod
    def findjobs(jobid=None,jobname=None,channel=""):
        # channel = "", "-d", "-p", "-r"
        if jobid is None and jobname is None:
            cmd = 'bjobs ' + channel + ' | grep ps'
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: process.kill()
            except: pass
            process.wait() 
        elif jobid is not None and jobname is None:
            cmd = "bjobs " + channel + " | grep " + str(jobid)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: process.kill()
            except: pass
            process.wait() 
        elif jobid is None and jobname is not None:
            cmd = 'bjobs -J ' + '*\"' + jobname + '\"*' + ' ' + channel + ' | grep ps'
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: process.kill()
            except: pass
            process.wait() 
        elif jobid is not None and jobname is not None:
            cmd = 'bjobs -J ' + '*\"' + jobname + '\"*' + ' ' + channel + ' | grep ' + str(jobid)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            try: process.kill()
            except: pass
            process.wait()
        return out
    
    @staticmethod
    def killjobs(jobid):
        cmd = "bkill " + str(jobid)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out,err = process.communicate()
        try: process.kill()
        except: pass
        process.wait()
        return not err 

    @staticmethod
    def jobname2ids(jobname):
        jobids = []
        out = WorkerBsubPsana.findjobs(jobname=jobname)
        if not isinstance(out,str):
            return jobids
        for s in out.split("\n"):
            jobids.append(int(s.split()[0]))
        return jobids

    @staticmethod
    def status_jobname(jobname=None):
        # waiting,running,failed,terminated,done
        if jobname is None:
            return None
        # Check -d (done)
        out = WorkerBsubPsana.findjobs(jobname=jobname,channel="-d")
        partial_failed = False
        if "exit" in out:
            partial_failed = True
        ## check in incomplete jobs
        if partial_failed:
            return "failed"
        out = WorkerBsubPsana.findjobs(jobname=jobname)
        if len(out) == 0:
            return "done"
        return "running"

    @staticmethod
    def status_jobid(jobid=None):
        # waiting,running,failed,terminated,done,pending,suspended
        if jobid is None:
            return None 
        out = WorkerBsubPsana.findjobs(jobid=jobid) # None/RUN/PEND/SUS
#         print "find jobid bjobs", out
        if len(out) == 0:
            out3 = WorkerBsubPsana.findjobs(jobid=jobid,channel="-d")
#             print "find jobid bjobs -d", out3
            if len(out3) == 0:
                return "pending"
            if "done" in out3.lower():
                return "done"
            if "exit" in out3.lower():
                return "failed"
            return "terminated"
        if "susp" in out.lower():
            return "suspended"
        if "pend" in out.lower():
            return "pending"
        return "running"

    def status(self):
        if self.__status__ == "done":
            return "done"
        if not self.submited:
            return "waiting"
        if not self.submitus:
            return "failed"
        if self.p.poll() is None:
            return "running"
        elif self.p.poll() > 0:
            return "failed"
        elif self.p.poll() < 0:
            return "terminated"
        # it means p.poll() == 0: done
        if not hasattr(self,"jobid"):
            import re
            self.p.poll() 
            out,err = None,None
            try: out,err = self.p.communicate()
            except: pass
            self.submitout = self.submitout or out
            self.submiterr = self.submiterr or err
            jobid = re.findall("<(.*?)>",self.submitout)
            if len(jobid)==0:
                self.jobid = None
            else:
                self.jobid = int(jobid[0])
        if not hasattr(self,"jobname"):
            self.jobname = self.actpack.get("jobname") 
        if self.jobname:
            self.__status__ = WorkerBsubPsana.status_jobname(self.jobname)
            return self.__status__
        if self.jobid:
            self.__status__ = WorkerBsubPsana.status_jobid(self.jobid)
            return self.__status__
        return "noaccess"

    def ready(self):
        # waiting,running,pending,suspended,failed,terminated,done
        if self.status() not in ["running","waiting","pending","suspended"]:
            return True
        return False

    def success(self):
        if self.status() in ["done"]:
            return True
        return False

    def returned(self):
        if self.status() in [None,"running","waiting","pending","suspended"]:
            return {"return":None}
        elif self.status() in ["failed","terminate"]:
            out,err = None,None
            try: out,err = self.p.communicate()
            except: pass
            self.submitout = self.submitout or out
            self.submiterr = self.submiterr or err
            return {"out":self.submitout,"err":self.submiterr}
        elif self.status() in ["done"]:
            out,err = None,None
            try: out,err = self.p.communicate()
            except: pass
            self.submitout = self.submitout or out
            self.submiterr = self.submiterr or err
            return {"out":self.submitout,"err":self.submiterr}
        raise NotImplemented
    
    def jobreturn(self):
        if self.actpack.get("freturn") and self.status()=="done":
            import pickle
            return pickle.load(open(self.actpack.get("freturn"),"rb"))
        return None

    def wait(self):
        try: 
            out, err = self.p.communicate()
            self.submitout = out
            self.submiterr = err
        except: pass

    def close(self):
        WorkerBsubPsana.killjobs(self.jobid)
        try: self.p.terminate()
        except: pass
        try: self.p.kill()
        except: pass
        try: self.p.wait()
        except: pass

    def clear(self):
        self.close()
        for key in self.__dict__:
            setattr(self,key,None)

    def runtime(self):
        return time.time() - self.starttime
        
    def _queue(self):
        return self.actpack.get("queue") or "psanaq"

    def _cpus(self):
        return self.actpack.get("cpus")  or 1

    def _pnode(self):
        nodes = self.actpack.get("nodes") or 1
        return int(math.ceil(self._cpus() * 1.0 / nodes))

    def _flog(self):
        if self.actpack.get("flog") is not None:
            return self.actpack.get("flog")
        self.actpack["flog"] = os.path.realpath(''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) \
                                                    for _ in range(20)) + ".out")
        return self.actpack.get("flog")

    def makecmd(self):
        if self.actpack.get("actobj").get("flaunch"):
            command = 'bsub -q %s -x -n %d -R "span[ptile=%d]" -o %s mpirun python %s'%(self._queue(),\
                    self._cpus(), self._pnode(), self._flog(), self.actpack.get("actobj").get("flaunch"))
            return command
        else: return self.actpack.get("actobj").get("command")