import os, sys
import numpy as np
import h5py

from StreamFile import *

ss = iStream()
ss.initial('cxilp9515_zhen_laser_on_0050ns.stream')
ss.get_label()
ss.get_info()
realpeak = ss.get_realPeak()
predpeak = ss.get_predPeak()

ss.save_label('cxilp9515_laser_on_0050ns.cxi')
ss.save_info('cxilp9515_laser_on_0050ns.cxi')

ss.clear()
