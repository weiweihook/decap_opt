import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d, make_interp_spline
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import os

def delete_defore(filename,output):
    with open(filename) as f, open(output, 'w') as temp:
        for line in f:
            parts = line.split('=')
            if len(parts) > 1:
                temp.write(parts[1])
            else:
                temp.write(line)
    with open(output, 'r') as f:
        content = f.read()
        new_content = content.replace('p','')
    with open(output, 'w') as f:
        f.write(new_content)


if __name__ == "__main__":
    delete_defore('init_int_param_dcap.txt', 'init_param_dcap.txt')