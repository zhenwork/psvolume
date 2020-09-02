import copy
import os,sys 
from numba import jit
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

import core.utils
import core.fsystem
import diffuse.datafile

class ArgumentFromFile:
    def __init__(self,args):
        self.decode(args)

    def decode(self,args):
        # args is a list
        self.file_name = []
        self.file_type = []
        self.available_keys = []
        self.convert_keys = []
        self.file_to_keys = {}
        self.keys_to_file = {}
        self.origin_keys = {} 

        from_file_input = copy.deepcopy(args)
        if not isinstance(from_file_input,list):
            return 
        if len(from_file_input)==0:
            return  

        fixed_value_keys = []
        for idx, file_object in enumerate(from_file_input):
            if ")" in file_object:
                _name = os.path.realpath(file_object.split("(")[0])
                _type = file_object.split("(")[1].split(")")[0]
            else:
                _name = os.path.realpath(file_object.split(":")[0])
                _type = None

            _available_keys = diffuse.datafile.Fmanager.file_keys(_name,_type)
            _convert_keys = {}
            _keep_keys = []
            _reject_keys = []
            
            if ":" in file_object:
                for key_object in file_object.split(":")[1].split(","):
                    if ">" in key_object:
                        x,y = key_object.strip().split(">")
                        _convert_keys[x] = y
                    elif "@" in key_object:
                        _keep_keys.append(key_object.strip().split("@")[1])
                    elif "!" in key_object:
                        _reject_keys.append(key_object.strip().split("!")[1])

            for dst_key in _convert_keys.values():
                if dst_key in _available_keys:
                    _available_keys.remove(dst_key)

            for src_key in _available_keys:
                dst_key = _convert_keys.get(src_key) or src_key
                if (src_key in _keep_keys) and (src_key not in _reject_keys):
                    self.keys_to_file[dst_key] = {"file_name":_name,"file_type":_type}
                    fixed_value_keys.append(dst_key)
                    self.origin_keys[dst_key] = src_key
                elif src_key in _reject_keys:
                    pass 
                elif dst_key not in fixed_value_keys:
                    self.keys_to_file[dst_key] = {"file_name":_name,"file_type":_type}
                    self.origin_keys[dst_key] = src_key
                else:
                    pass 

            self.file_name.append(_name)
            self.file_type.append(_type)
            self.available_keys.append(_available_keys)
            self.convert_keys.append(_convert_keys)
        
        for dst_key in self.keys_to_file:
            file_name = self.keys_to_file[dst_key]["file_name"]
            if file_name not in self.file_to_keys:
                self.file_to_keys[file_name] = [dst_key]
                continue
            self.file_to_keys[file_name].append(dst_key)
        self.file_to_type = dict(zip(self.file_name,self.file_type))
        return 


class ArgumentIntoFile:
    def __init__(self,args):
        self.decode(args) 

    def decode(self,args):
        # args is a list
        self.file_name = []
        self.keep_keys = []
        self.reject_keys = []
        self.convert_keys = []

        from_file_input = copy.deepcopy(args)
        if not isinstance(from_file_input,list):
            return 
        if len(from_file_input)==0:
            return  

        for idx, file_object in enumerate(from_file_input):
            if ")" in file_object:
                _name = os.path.realpath(file_object.split("(")[0])
            else:
                _name = os.path.realpath(file_object.split(":")[0])

            _keep_keys = []
            _reject_keys = []
            _convert_keys = {}
            
            if ":" in file_object:
                for key_object in file_object.split(":")[1].split(","):
                    if ">" in key_object:
                        x,y = key_object.strip().split(">")
                        _convert_keys[x] = y
                    elif "@" in key_object:
                        _keep_keys.append(key_object.strip().split("@")[1])
                    elif "!" in key_object:
                        _reject_keys.append(key_object.strip().split("!")[1])

            self.file_name.append(_name) 
            self.keep_keys.append(_keep_keys) 
            self.reject_keys.append(_reject_keys) 
            self.convert_keys.append(_convert_keys) 

        return 

class ArgumentKeyParams:
    def __init__(self,args):
        self.decode(args)

    def decode(self, args):
        # get variables of an action 
        # args = ["x=10;y=20","z=30"]
        self.params = {}
        if not isinstance(args,list):
            return 
        if len(args)==0:
            return 

        for one_args in args:
            if ";" in one_args:
                one_args = one_args.split(";")
            else:
                one_args = [one_args]
            for key_val in one_args:
                key,val = key_val.split("=")
                if val.lower()=="none":
                    self.params[key] = None
                elif val.lower()=="true":
                    self.params[key] = True
                elif val.lower()=="false":
                    self.params[key] = False
                else:
                    try:
                        self.params[key] = int(val)
                    except:
                        try:
                            self.params[key] = float(val)
                        except:
                            self.params[key] = val
        return 


class Image:
    def _init__(self,**kwargs):
        self.changed_keys = set([])
        for key in kwargs:
            setattr(self,key,kwargs[key])

    def initialize(self):
        if not hasattr(self,"image"):
            if hasattr(self,"image_file"):
                data = core.fsystem.CBFmanager.reader(self.image_file)
                self.__dict__.update(data)
        if not hasattr(self,"backg"):
            if hasattr(self,"backg_file"):
                data = core.fsystem.CBFmanager.reader(self.backg_file)
                self.__dict__.update(data)
        if 
        

    def apply_detector_mask(self,**kwargs):
        self.changed_keys += set(["image"])
        return 

    def remove_bad_pixels(self,**kwargs):
        self.changed_keys += set(["image","mask"])
        return 

    def subtract_background(self,**kwargs):
        self.changed_keys += set(["image","mask"])
        return 

    def parallax_correction(self,**kwargs):
        return 

    def polarization_correction(self,**kwargs):
        self.changed_keys += set(["image"])
        return 

    def solid_angle_correction(self,**kwargs):
        self.changed_keys += set(["image"])
        return 

    def detector_absorption_correction(self,**kwargs):
        self.changed_keys += set(["image"])
        return 

    def remove_bragg_peak(self,**kwargs):
        self.changed_keys += set(["image"])
        return 

    def calculate_radial_profile(self,**kwargs):
        self.changed_keys += set(["radial_profile"])
        return 

    def calculate_overall_intensity(self,**kwargs):
        self.changed_keys += set(["overall_intensity"])
        return 

    def calculate_average_intensity(self,**kwargs):
        self.changed_keys += set(["average_intensity"])
        return 

    def changed_params(self):
        changed_data = {}
        for key in self.changed_keys:
            changed_data[key] = getattr(self,key)
        return changed_data

    def free_memory(self):
        for key in self.__dict__.keys():
            delattr(self, key)


class ImportManager:
    """
    parser.add_argument("--image_file", help="cbf",default=None,nargs="*") 
    parser.add_argument("--backg_file", help="cbf",default=None,nargs="*") 
    parser.add_argument("--dials_report_file", help="file",default=None,type=str) 
    parser.add_argument("--dials_expt_file", help="file",default=None,type=str) 
    parser.add_argument("--gxparms_file", help="file",default=None,type=str) 
    parser.add_argument("--from_file", help="file_name_*.h5py(h5py):data>image,back>backg,!scaler,@Amat_0_invA,@ hello.npy(numpy):data>scaler",\
                        default=None,nargs="*") 
    parser.add_argument("--into_file", help="file_name_*.h5py(h5py):data>image,back>backg,!scaler,@Amat_0_invA,@ hello.npy(numpy):data>scaler",\
                        default="./diffuse.dat",nargs="*") 
    """
    def __init__(self, args):
        self.args = args.copy()

    def prepare_required_params(self): 
        input_manager = ArgumentFromFile(self.args.get("from_file"))
        self.file_to_keys = input_manager.file_to_keys
        self.keys_to_file = input_manager.keys_to_file
        self.file_to_type = input_manager.file_to_type
        self.origin_keys = input_manager.origin_keys

    def start_process(self):
        self.params = {}
        if self.args.get("image_file") not in [None,[]]:
            self.params = core.utils.dict_merge(self.params,{"image_file":self.args.get("image_file")})
        if self.args.get("backg_file") not in [None,[]]:
            self.params = core.utils.dict_merge(self.params,{"backg_file":self.args.get("backg_file")})
        if self.args.get("dials_report_file") not in [None,[]]:
            data = diffuse.datafile.load_file(self.args.get("dials_report_file"),file_type="dials_report")
            self.params = core.utils.dict_merge(self.params,data)
        if self.args.get("dials_expt_file") not in [None,[]]:
            data = diffuse.datafile.load_file(self.args.get("dials_expt_file"),file_type="dials_expt")
            self.params = core.utils.dict_merge(self.params,data)
        if self.args.get("gxparms_file") not in [None,[]]:
            data = diffuse.datafile.load_file(self.args.get("gxparms_file"),file_type="gxparms")
            self.params = core.utils.dict_merge(self.params,data)

        if getattr(self, "file_to_keys") in [None, {}]:
            return 

        for file_name in self.file_to_keys:
            file_type = self.file_to_type[file_name]
            keep_keys = [self.origin_keys[key] for key in self.file_to_keys[file_name]]
            data = diffuse.datafile.load_file(file_name=file_name,\
                                              file_type=file_type,\
                                              keep_keys=keep_keys)
            for from_key,into_key in self.input_convert_keys[idx]:
                if from_key in data.keys():
                    data[into_key] = data.pop(from_key)
            self.params = core.utils.dict_merge(self.params,data)
            data = None

    def update_result(self):
        if self.args.get("into_file") is None:
            return 

        if ":" not in self.args.get("into_file"):
            file_name = self.args.get("into_file").strip()
            core.fsystem.PVmanager.modify(self.params, file_name)
            return 

        file_name = self.args.get("into_file").split(":")[0]
        result = {}
        if "@" in self.args.get("into_file"):
            keep_keys = []
            for key_object in self.args.get("into_file").split(":")[1].split(","):
                if "@" in key_object:
                    keep_keys.append(key_object.strip().split("@")[1])
            for key in keep_keys:
                result[key] = self.params.get(key)

        if "!" in self.args.get("into_file"):
            reject_keys = []
            for key_object in self.args.get("into_file").split(":")[1].split(","):
                if "!" in key_object:
                    reject_keys.append(key_object.strip().split("!")[1])
            for key in reject_keys:
                result.pop(key)

        if ">" in self.args.get("into_file"):
            convert_keys = []
            for key_object in self.args.get("into_file").split(":")[1].split(","):
                if ">" in key_object:
                    convert_keys.append(tuple(key_object.strip().split(">")))
            for x,y in convert_keys:
                if x in result:
                    result[y] = result.pop(x)

        core.fsystem.PVmanager.modify(result, file_name)
        return 

    def free_memory(self):
        for key in self.__dict__.keys():
            delattr(self, key)


class ImageProcessMaster:
    @staticmethod
    class status_params:
        def __init__(self,status=False,params={}):
            self.status = status
            self.params = copy.deepcopy(params)

    def __init__(self,args):
        self.args = args.copy()
        self.accept_variables = ["image_file","backg_file","image","mask",\
                                 "pixel_size_um", "detector_distance_mm",\
                                 "detector_center_px","wavelength_A","detector_thickness_um",\
                                 "Amat_0_invA", "Amat_invA", "Bmat_invA", "rotation_angle_deg",\
                                 "detector_absorption_rate_um","per_image_multiply_scaler",\
                                 "polarization_fr","lattice_constant_A_deg"]

        self.accept_processes = ["apply_detector_mask",
                                 "remove_bad_pixels",
                                 "subtract_background",
                                 "parallax_correction",
                                 "polarization_correction",
                                 "solid_angle_correction",
                                 "detector_absorption_correction",
                                 "remove_bragg_peak",
                                 "calculate_radial_profile",
                                 "calculate_overall_intensity",
                                 "calculate_average_intensity"]

    def prepare_input_params(self):
        for action in self.args.keys():
            if action == "from_file":
                input_manager = ArgumentFromFile(self.args.get("from_file"))
                self.i_file_to_keys = input_manager.file_to_keys
                self.i_keys_to_file = input_manager.keys_to_file
                self.i_file_to_type = input_manager.file_to_type
                self.i_origin_keys  = input_manager.origin_keys
                input_manager = None
                continue
            elif action == "into_file":
                output_manager = ArgumentIntoFile(self.args.get("into_file"))
                self.o_file_name = output_manager.file_name 
                self.o_keep_keys = output_manager.keep_keys
                self.o_reject_keys = output_manager.reject_keys
                self.o_convert_keys = output_manager.convert_keys
                output_manager = None
                continue 
            setattr(self, action, status_params(status=True,ArgumentKeyParams(self.args.get(action))))
        for process in self.accept_processes:
            if not hasattr(self,process):
                setattr(self,process, status_params(status=False,{}))

    def request_image_data(self,idx=0):
        image_data = {}
        for file_name in self.i_file_to_keys:
            file_type = self.i_file_to_type[file_name]
            keys = copy.deepcopy(self.i_file_to_keys[file_name])
            for accept_key in self.accept_variables:
                if accept_key in keys:
                    if accept_key=="image":
                        image_data["image"] = core.fsystem.H5manager.reader(idx=0)
                    else:
                        image_data[accept_key] = diffuse.datafile.load_file(file_name)
        return Image(**image_data)

    def request_backg_data(self,idx=0):
        backg_data = {}
        for file_name in self.i_file_to_keys:
            file_type = self.i_file_to_type[file_name]
            keys = copy.deepcopy(self.i_file_to_keys[file_name])
            for accept_key in self.accept_variables:
                if accept_key in keys:
                    if accept_key=="backg"::
                        backg_data["backg"] = core.fsystem.H5manager.reader(idx=0)
                    else:
                        backg_data[accept_key] = diffuse.datafile.load_file(file_name) 
        return Image(**backg_data)

    def update_result(self,idx=None,changed_image_data=None):
        core.fsystem.PVmanager.modify(changed_image_data,idx=idx) 

    def free_memory(self):
        for key in self.__dict__.keys():
            delattr(self, key)


class MergeManager:
    def __init__(self,args):
        self.args = args.copy()
        self.accept_variables = ["image_file","image","mask",\
                                 "pixel_size_um", "detector_distance_mm",\
                                 "detector_center_px","wavelength_A",\
                                 "Amat_0_invA", "Amat_invA", "Bmat_invA", "rotation_angle_deg",\
                                 "per_image_multiply_scaler","lattice_constant_A_deg"]

    def prepare_input_params(self):
        # get required files
        self.ipm = workflow.manager.ImageProcessMaster(args=self.args)
        self.ipm.accept_variables = self.accept_variables
        self.ipm.prepare_required_params() 

    def merge_with_averaging(self,**kwargs):
        volume = None 
        weight = None 
        if kwargs.get("select_pattern_idx") is None:
            kwargs["select_pattern_idx"] = np.arange(self.ipm.num_images)
        else:
            kwargs["select_pattern_idx"] = workflow.utils.get_idx_list(kwargs["select_pattern_idx"])

        for image_idx in kwargs["select_pattern_idx"]:
            image_obj = ipm.request_image_data(idx=image_idx).__dict__
            core.utils.merge_dict(image_obj, kwargs)
            volume, weight = diffuse.mapper.image_2d_to_volume_3d(**image_obj)

        self.volume = volume
        self.weight = weight


class ScalerManager:
    def __init__(self,args):
        self.args = args.copy()
        self.accept_variables = ["image_file","image","mask", "dials_", "dials_", \
                                 "per_image_multiply_scaler","average_intensity","overall_intensity"]

    def prepare_required_params(self):
        self.ipm = workflow.manager.ImageProcessMaster(args=self.args)
        self.ipm.accept_variables = self.accept_variables
        self.ipm.prepare_required_params() 

    def calculate_per_image_multiply_scaler(self):
        # 
        if imp.calculate_per_image_multiply_scaler.status:
            


class PCAManager:
    def __init__(self,args):
        pass 

    def pca_calculation(self):
        return 

    def pca_normalization(self,num_pca=3,divide=True):
        self.pca_background = 
        self.changed_keys += set(["image"]) 
        return 

    def update_result(self):
        return 

class VolumeManager:
    def __init__(self,args):
        pass 

    def laue_symmetrization_background_subtraction(self,symmetry=1,**kwargs):
        return 

    def background_subtraction_laue_symmetrization(self,symmetry=1,**kwargs):
        return 

    def friedel_symmetrization_background_subtraction(self,**kwargs):
        return 

    def background_subtraction_friedel_symmetrization(self,**kwargs):
        return 

    @staticmethod
    def calculate_radial_background(data,mask,**kwargs):
        return 

    @staticmethod
    def laue_symmetrization(data,mask,**kwargs):
        return 

    @staticmethod
    def friedel_symmetrization(data,mask,**kwargs):
        return 

