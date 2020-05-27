# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:07:24 2020

@author: lxh4557
"""

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
sns.set()
tf.compat.v1.random.set_random_seed(1234)

df = pd.read_csv('../dataset/GOOG-year.csv')
df.head()