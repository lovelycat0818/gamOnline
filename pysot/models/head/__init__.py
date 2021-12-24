# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN
from pysot.models.head.gam import GAM_Attention, GAMAllLayer
from pysot.models.head.gam_eca import GAM_eca_Attention, GAMecaAllLayer


RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN,
       }
GAM = {
        'GAM_s': GAM_Attention,
        'GAM_m': GAMAllLayer,
        'GAM_e': GAMecaAllLayer
}


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)

def get_gam(name, **kwargs):
    return GAM[name](**kwargs)

