#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    10/07/2017
""" 

import numpy as np

import camerainterface


if __name__=='__main__':
    ci = camerainterface.ProductIdentifier()
    ci.start()
