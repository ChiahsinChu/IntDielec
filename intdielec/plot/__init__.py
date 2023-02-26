import os
import matplotlib.pyplot as plt

# from .core import *

def use_style(style_name):
    suffix="mplstyle"
    fname = "%s.%s" % (style_name, suffix)
    fname = os.path.join(__path__[0], "mplstyle", fname)
    try:
        plt.style.use(fname)
    except:
        print("Warning: no style %s is found. Use matplotlib default style." % style_name)