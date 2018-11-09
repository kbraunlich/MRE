#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:32:53 2018

@author: kurtb
"""

from PIL import Image
import glob
import os

#%%
f = '/home/kurtb/Dropbox/code/multiple_relation_embed/coil-100'
outf = '/home/kurtb/Dropbox/code/multiple_relation_embed/coil-100_grey'

ps = glob.glob(f+'/*.png')
for p in ps:
    n=os.path.basename(p)
    img = Image.open(p).convert('LA')
    img.save(outf+'/'+n)
    
#%%