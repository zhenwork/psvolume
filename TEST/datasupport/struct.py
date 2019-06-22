"""
1. Use standard polarization label: 
    *p=1 means y polarization
    *p=-1 means x polarization
    *p can be a float number in [-1,1]
2. Once pixel_size is set, pixel_size_x and pixel_size_y will be ignored
3. Params: phi and rotation_axis are specifically used in ratotion measurement
4. Amat and Bmat are in the published standard as CrystFel

"""


class struct(object):
    def __init__(self):
        self.image = None
        self.mask = 1
        self.exposure_time_sec = None
        self.phi_deg = None
        self.wavelength_A = None
        self.pixel_size_mm = None
        self.x_pixel_size_mm = None
        self.y_pixel_size_mm = None
        self.polarization = -1
        self.detector_distance_cm = None
        self.beam_x_pix = None
        self.beam_y_pix = None
        self.lattice_constant_nm_deg = None
        self.scale = 1.0
        self.rotation_axis = "x"
        self.rotation_mat = None
        self.Amat_invnm = None
        self.Bmat_invnm = None