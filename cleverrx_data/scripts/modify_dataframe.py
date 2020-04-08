#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:08:27 2020

@author: djweber3
"""

import pandas as pd
import numpy as np 

def view_and_change_columns(df, view_columns, change_column): 
    
    for i in df.index: 
        print(df.loc[i, view_columns])
        u_input = input("relevance value?")
        df.loc[i, change_column] = int(u_input) 
    
    return df 
    
