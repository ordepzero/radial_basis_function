# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:00:20 2016

@author: PeDeNRiQue
"""

import numpy as np

def read_file(filename, separator):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(separator)])   
    return np.array(array);
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data
    
def str_to_number(data):
    return[[float(j) for j in i] for i in data]