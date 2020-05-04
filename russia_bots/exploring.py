import pandas as pd
import numpy as np
import matplotlib
from scipy import stats
import matplotlib.pyplot as plt
#%%

fake_data = pd.DataFrame([[1,2,3],[1,2,3],[1,2,3]])
fake_data

stats.zscore(fake_data)
