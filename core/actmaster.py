class ActBase(object):
    def actpath(self): 
        return os.path.realpath(os.path.abspath(__file__))
    def run(self): 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        global size,rank
        rank = comm.Get_rank()
        size = comm.Get_size() 
        

class ActMaster:
    def __init__(self,state=None,action=None,workdir="./"): 
        self.action  = action
        self.status  = "waiting"
        self.codedir = os.path.dirname(os.path.abspath(__file__))
        self.workdir = os.path.realpath(workdir)
        self.datadir = os.path.join(self.workdir,"psdiffuse")

    def prepare(self):
        assert os.path.isdir(self.workdir)
        assert os.path.isdir(self.codedir)
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)
        if not os.path.isdir(self.datadir+"/data"):
            os.makedirs(self.datadir+"/data") 

    def update(self):
        if hasattr(self,"worker"):
            self.status = self.worker.status()
        if not hasattr(self,"result"):
            self.result = {}
        if self.status in ["done","failed","terminated"]:
            self.result.update(self.worker.returned())
        return self

    def start(self):
        assert self.prepare()
        assert self.actfind()
        self.worker = self.toworker()
        self.worker.start()
        
    def communicate(self):
        self.update()
        self.wait()
        self.close()
        self.update()
        if self.pipe is not None:
            self.pipe.put(self.feedback())
        else: return self.feedback()

    def returndump(self):
        self.update()
        if self.actpack.get("freturn"):
            import pickle
#             print self.result, self.actpack.get("freturn")
            pickle.dump(self.result,open(self.actpack.get("freturn"),"wb"))
    
    def returnload(self): 
        if self.actpack.get("freturn"):
            import pickle 
            return pickle.load(open(self.actpack.get("freturn"),"rb"))
        return None
    
    def actfind(self,actobj=None):
        # actobj/self.actpack None,str,dict,obj
        if actobj is None:
            actobj = self.actpack.get("actobj")
        if actobj is None:
            return False
        elif isinstance(actobj, str):
            # actobj = "ImageMask"
            paths = []
            paths.append(os.path.join(self.workdir,"action.py"))
            paths.append(os.path.join(self.datadir,"./code/action.py"))
            for path in paths:
                if not os.path.isfile(path):
                    continue
                import imp
                obj = imp.load_source('', path)
                if hasattr(obj,actobj):
                    self.actpack.update({"actobj":{"path":path, "name":actobj}})
                    return True
            return False
        elif isinstance(actobj,dict):
            if actobj.get("command") is not None:
                return True
            if actobj.get("pickle") is not None:
                if os.path.isfile(actobj.get("pickle")):
                    return True
                return False
            if actobj.get("flaunch") is not None:
                if os.path.isfile(actobj.get("flaunch")):
                    return True
                return False
            if os.path.isfile(actobj.get("path")):
                import imp
                obj = imp.load_source('', actobj.get("path"))
                if hasattr(obj,actobj.get("name")):
                    return True
            return False
        return True

    def toworker(self):
        assert self.actfind()
        objtype,workmode = self.actmode()
        worker = Worker(mode=workmode)
        actpack = self.actpolish()
        worker.__init__(actpack)
        return worker

    def actpolish(self):
        # obj,str,name,pickle,flaunch,command
        # transfer: "bsub"
        assert self.actfind()
        actpack = copy.deepcopy(self.actpack)
        dumpack = {"actobj":self.actpack.get("actobj"),\
                   "params":self.actpack.get("params")}
        actobj  = actpack.get("actobj")
        objtype,workmode = self.actmode()
        if workmode=="bsub":
            if objtype == "flaunch":
                pass
            elif (objtype == "command") and ("bsub" in actobj.get("command")):
                pass
            else:
                dumpack["envpath"] = actpack.get("envpath") or (sys.path + [self.codedir])
                dumpack["freturn"] = actpack.get("freturn") or os.path.join(self.datadir,"code/"+self.randomstr(20) + ".pkl")
                self.actpack.update({"freturn":dumpack["freturn"]})
                flaunch = self.actpack2flaunch(dumpack)
                actpack.update({"actobj":{"flaunch":flaunch},"freturn":dumpack["freturn"]})
        if workmode=="backg":
            if objtype in ["obj","str","name","pickle"]:
                dumpack["envpath"] = actpack.get("envpath") or (sys.path + [self.codedir])
                dumpack["freturn"] = actpack.get("freturn") or os.path.join(self.datadir,"code/"+self.randomstr(20) + ".pkl")
                self.actpack.update({"freturn":dumpack["freturn"]})
                flaunch = self.actpack2flaunch(dumpack)
                actpack.update({"actobj":{"flaunch":flaunch},"freturn":dumpack["freturn"]})
        return actpack

    def actpack2flaunch(self,actpack):
        # 1. save the actpack into file factpack
        # 2. write flaunch, it can load factpack
        # MIND: the actpack should shift from bsub into function/command
        # TODO: maybe save a new Master(actpack) into file
        import pickle 
        import dill
        factpack = os.path.join(self.datadir,"code/"+self.randomstr(20) + ".pkl")
        dill.settings['recurse'] = True 
        pickle.dump(actpack,open(factpack,"wb"))
        codeline = \
"""
master.returndump()
"""%factpack
        flaunch = os.path.join(self.datadir,"code/"+self.randomstr(20) + ".py")
        with open(flaunch,"w") as f:
            f.write(codeline); f.write("\n");
        return flaunch

    def takenote(self,notes):
        # notes: dict/str  ==> or list of (dict/str)
        if isinstance(notes,(list,tuple,set)):
            for note in notes:
                self.takenote(note)
        elif isinstance(notes,dict):
            if not hasattr(self,"logger"):
                self.logger = {}
            self.logger.update(notes)
            return True
        elif isinstance(notes,str):
            if not hasattr(self,"logger"):
                self.logger = {"notes":[]}
            elif self.logger.get("notes") is None:
                self.logger["notes"] = []
            else:
                pass
            self.logger["notes"].append(notes)
            return True
        else:
            raise NotImplemented
            
    def release(self):
        self.update()
        return 

    def todict(self):
        self.update()
        return self.__dict__

    def feedback(self):
        self.update()
        return self.todict()
    
    def wait(self):
        while self.update().status not in ["done","failed","terminated"]:
            time.sleep(2)
        self.update()
        
    def close(self):
        if hasattr(self,"worker"):
            self.worker.close()

    def actmode(self):
        
        return objtype, workmode