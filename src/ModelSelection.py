from keras.src.layers import Activation
from pyhdf.SD import SD, SDC
import h5py
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress
from scipy.stats import t
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import compute_class_weight
from statsmodels.tsa.seasonal import STL
from keras.utils import to_categorical
from keras.src.callbacks import EarlyStopping
import xgboost as xgb
from keras.models import Sequential
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import ConvLSTM2D, Dense, Flatten, TimeDistributed, LSTM, Conv1D, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Dropout
from keras.optimizers.legacy import Adam
from keras.regularizers import l2
from keras.layers import BatchNormalization, LeakyReLU, ReLU
from keras.src.optimizers import RMSprop

from src.ForestHealthClassification import ForestHealthClassification
from src.ModelManager import ModelManager


class ModelSelection:
    def __init__(self, sg_values):
        self.sg_values = None
        pass

    def model_comparison(self):
        sg_h5 = f'./sg.h5'
        if os.path.exists(sg_h5):
            print(f"Read SG values from {sg_h5}")
            t1 = datetime.now()
            with h5py.File(sg_h5, 'r') as hf:
                self.sg_values = hf['SG'][:]
            ModelManager.print_run_time(t1)
        else:
            print("File not found: ", sg_h5)
            exit(123)

        # mm_lr = ModelManager(self.sg_values, is_discrete=True)
        # mm_lr.train_lr(is_plot=True)
        # result_lr = mm_lr.results_dic["metrics"]
        #
        # mm_xgb = ModelManager(self.sg_values)
        # mm_xgb.train_xgb(is_plot=True)
        # result_xgb = mm_xgb.results_dic["metrics"]
        #
        # mm_dt = ModelManager(self.sg_values)
        # mm_dt.train_dt(is_plot=True)
        # result_dt = mm_dt.results_dic["metrics"]

        # ms_nn1_1 = ModelManager(self.sg_values, is_discrete=True)
        # ms_nn1_1.train_nn(1, (32, 32, 32),
        #                   (128, 64), (0.3, 0.3),
        #                   (128, 64), (0.3, 0.3),
        #                   1, 0.5, 0.002, 0.01, is_plot=True)

        ms_nn4 = ModelManager(self.sg_values)
        ms_nn4.train_nn(4, (8, 8, 8),
                        (8, 8), (0.3, 0.3),
                        (128, 128), (0.3, 0.3),
                        1, 0.5, 0.001, 0.01, is_plot=True)

        # done
        # ms_nn2_1 = ModelManager(self.sg_values, is_discrete=True)
        # ms_nn2_1.train_nn(2, (32, 32, 32),
        #                   (128, 64), (0.6, 0.4),
        #                   (128, 64), (0.6, 0.4),
        #                   2, 0.5, 0.005, is_plot=True)

        # doing
        # ms_nn2_2 = ModelManager(self.sg_values, is_discrete=True)
        # ms_nn2_2.train_nn(2, (32, 32, 32),
        #                   (64, 32), (0.6, 0.4),
        #                   (64, 64), (0.6, 0.4),
        #                   2, 0.5, 0.005, is_plot=True)

        # ms_nn3_1 = ModelManager(self.sg_values, is_discrete=True)
        # ms_nn3_1.train_nn(3, (12, 12, 12),
        #                 (64, 64), (0.6, 0.4),
        #                 (64, 64), (0.6, 0.4),
        #                 1, 0.5, 0.2, 0.01, is_plot=True)

        # ms_nn3_2 = ModelManager(self.sg_values, is_discrete=True)
        # ms_nn3_2.train_nn(3, (12, 12, 12),
        #                 (32, 32), (0.6, 0.4),
        #                 (64, 64), (0.6, 0.4),
        #                 1, 0.5, 0.05, is_plot=True)

        # ms_nn3_3 = ModelManager(self.sg_values, is_discrete=True)
        # ms_nn3_3.train_nn(3, (12, 12, 12),
        #                 (16, 16), (0.6, 0.4),
        #                 (64, 64), (0.6, 0.4),
        #                 1, 0.5, 0.05, is_plot=True)

        # self.foo(ms_nn2_2)

        return

    @staticmethod
    def foo(model_manager):
        y_true, y_pred = model_manager["y_test_label"], model_manager["y_pred_label"]

        from sklearn.metrics import accuracy_score, f1_score
        print(accuracy_score(y_true, y_pred))
        print(f1_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))

        # from sklearn.metrics import RocCurveDisplay, roc_curve
        # fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        #
        # from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
        # prec, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
        # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # roc_display.plot(ax=ax1)
        # pr_display.plot(ax=ax2)

        return
