#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:32:53 2018

@author: kurtb
"""

from PIL import Image
import glob
import os
import clarte as cl

#%%
f = '/home/kurtb/Dropbox/code/multiple_relation_embed/coil-100'
outf = '/home/kurtb/Dropbox/code/multiple_relation_embed/coil-100_grey'
#%%
ps = glob.glob(f+'/*.png')
for p in ps:
    n=os.path.basename(p)
    img = Image.open(p).convert('LA')
    img.save(outf+'/'+n)
    
#%% rename images
ps = glob.glob(outf+'/*.png')
for p in ps:
    n = os.path.basename(p)
    numb = (n[n.find('__')+2:n.find('.pn')])
    if len(numb)<3:
        newnumb = '%.3d'%int(numb)
        newn = n.replace(numb,newnumb)
#        cl.keyboard()
        os.rename(p,outf+'/%s'%newn)
    
    