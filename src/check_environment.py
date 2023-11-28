import sys

import keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform

'''
Python Platform: macOS-14.1.1-arm64-arm-64bit
Tensor Flow Version: 2.15.0
Keras Version: 2.15.0

Python 3.11.5 (main, Sep 11 2023, 08:31:25) [Clang 14.0.6 ]
Pandas 2.1.3
Scikit-Learn 1.3.2
SciPy 1.11.3
GPU is available
'''

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
