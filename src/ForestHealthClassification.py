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


class ForestHealthClassification:
    def __init__(self, clf_name, sg_values, metrics, years, lr_range=7):
        self.clf_name = clf_name
        self.sg_values = sg_values
        self.metrics = metrics
        self.years = years
        self.lr_range = lr_range
        self.yrs = self.years[-self.lr_range:]
        self.year_range = "%s~%s" % (self.yrs[-self.lr_range], self.yrs[-1])
        self.rootpath = '../h12v04'
        self.p_val_nan = 10
        self.std_errs = None
        self.p_values = None
        self.intercepts = None
        self.slopes = None
        self.r_values = None
        self.pixel_classes = None

    def linear_regress(self):
        sgs = self.sg_values[-self.lr_range:, :, :]
        years_col = self.yrs[:, np.newaxis, np.newaxis]
        x_mean = np.mean(years_col)
        y_mean = np.mean(sgs, axis=0)

        num = np.sum((years_col - x_mean) * (sgs - y_mean), axis=0)
        den = np.sum((years_col - x_mean) ** 2)

        # Calculate slope and intercept
        self.slopes = num / den
        self.intercepts = y_mean - self.slopes * x_mean

        # Calculate covariance and variance for r_value calculation
        covariance = np.sum((self.yrs[:, np.newaxis, np.newaxis] - x_mean) * (sgs - y_mean), axis=0)
        variance_x = np.sum((self.yrs - x_mean.squeeze()) ** 2)
        variance_y = np.sum((sgs - y_mean) ** 2, axis=0)
        denominator = variance_x * variance_y
        self.r_values = np.where(variance_y != 0, covariance / np.sqrt(denominator), 0)

        # Calculate p_value and std_err
        n = len(self.yrs)
        t_stat = self.r_values * np.sqrt(n - 2) / np.sqrt(1 - self.r_values ** 2)

        # p value
        df = n - 2
        self.p_values = t.sf(np.abs(t_stat), df) * 2

        # std err
        self.std_errs = np.sqrt((1 - self.r_values ** 2) * np.var(sgs, axis=0) / (n - 2))

    def classify_pixels(self):
        is_forest_growth = self.slopes[:, :] >= 0
        is_forest_decline = self.slopes[:, :] < 0

        # mark water region
        p_val = np.where(self.sg_values[-1, :, :] == 0, self.p_val_nan, self.p_values)

        self.pixel_classes = np.zeros_like(p_val, dtype=np.int8)

        self.pixel_classes[is_forest_decline & (p_val <= 0.05)] = 1
        self.pixel_classes[(is_forest_decline & (p_val > 0.05)) | is_forest_growth] = -1
        self.pixel_classes[p_val == self.p_val_nan] = -1

    def classify_pixels_33(self):
        is_forest_growth = self.slopes[:, :] >= 0
        is_forest_decline = self.slopes[:, :] < 0

        # mark water region
        p_val = np.where(self.sg_values[-1, :, :] == 0, self.p_val_nan, self.p_values)

        self.pixel_classes = np.zeros_like(p_val, dtype=np.int8)

        self.pixel_classes[is_forest_growth & (p_val <= 0.001)] = -3
        self.pixel_classes[is_forest_decline & (p_val <= 0.001)] = 3

        self.pixel_classes[is_forest_growth & (p_val > 0.001) & (p_val <= 0.01)] = -2
        self.pixel_classes[is_forest_decline & (p_val > 0.001) & (p_val <= 0.01)] = 2

        self.pixel_classes[is_forest_growth & (p_val > 0.01) & (p_val <= 0.05)] = -1
        self.pixel_classes[is_forest_decline & (p_val > 0.01) & (p_val <= 0.05)] = 1

        self.pixel_classes[p_val == self.p_val_nan] = -4

        self.pixel_classes[(p_val <= 1) & (p_val > 0.05)] = 0

        # pth = f"{self.rootpath}/data_training"
        # if not os.path.exists(pth):
        #     os.makedirs(pth)
        # year_range = "%s~%s" % (start_year, end_year)
        # np.save(f'{pth}/sg_clf_training_data_{year_range}.npy', self.pixel_classes)
        # return year_range

    def __plot_it(self, year_range):
        fig, ax = plt.subplots(figsize=(10, 10))
        norm = Normalize(vmin=-1.1, vmax=1.1)
        cax = ax.imshow(self.pixel_classes, norm=norm)

        major_ticks = np.arange(0, self.pixel_classes.shape[1] + 1, 1000)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        minor_ticks = np.arange(0, self.pixel_classes.shape[1] + 1, 200)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(axis='both', which='minor', size=0)

        ax.grid(True, which='both', color='white', linewidth=0.5, linestyle='--')
        ax.grid(which='minor', color='white', alpha=0.4)
        ax.grid(which='major', color='white', alpha=0.8)

        cbar = fig.colorbar(cax, orientation='vertical')
        ttl = (f'Forest Health prediction based on SG trend({self.clf_name}) {year_range}\n' +
               f'Prediction Accuracy: %s(MAE), %s(RMSE), %s(R2)' % (
                   "\'N/A\'" if self.metrics is None else f"%.2f" % self.metrics["Mean Absolute Error (MAE)"],
                   "\'N/A\'" if self.metrics is None else f"%.2f" % self.metrics["Root Mean Squared Error (RMSE)"],
                   "\'N/A\'" if self.metrics is None else f"%.2f" % self.metrics["R-Squared (R2)"])
               )
        ax.set_title(ttl)
        return fig

    def __plot_it_7(self, year_range):
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['#1E1EFF',  # Aquatic Area
                  '#EF6FFF',  # Strong Growth
                  '#7FEFBF',  # Significantly Growing
                  '#0FEF0F',  # Growing
                  '#9F9F9F',  # Stable
                  '#FFF0A5',  # Declining
                  '#FF6030',  # Significantly Declining
                  '#8F6030',  # Severely Declining
                  '#FFFFFF']  # empty
        cmap_custom = ListedColormap(colors)
        norm = Normalize(vmin=-4.5, vmax=4.5)
        cax = ax.imshow(self.pixel_classes, cmap=cmap_custom, norm=norm)

        major_ticks = np.arange(0, self.pixel_classes.shape[1] + 1, 1000)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        minor_ticks = np.arange(0, self.pixel_classes.shape[1] + 1, 200)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(axis='both', which='minor', size=0)

        ax.grid(True, which='both', color='white', linewidth=0.5, linestyle='--')
        ax.grid(which='minor', color='white', alpha=0.4)
        ax.grid(which='major', color='white', alpha=0.8)

        ticks_dict = {
            4: "_",
            3: "Severely Declining",
            2: "Significantly Declining",
            1: "Declining",
            0: "Stable",
            -1: "Growing",
            -2: "Significantly Growing",
            -3: "Strong Growth",
            -4: "Aquatic Area"
        }
        cbar = fig.colorbar(cax, ticks=list(ticks_dict.keys()), orientation='vertical')
        cbar.ax.set_yticklabels(ticks_dict.values())
        ttl = (f'Forest Health prediction based on SG trend({self.clf_name}) {year_range}\n' +
               f'Prediction Accuracy: %s(MAE), %s(RMSE), %s(R2)' % (
                   "\'N/A\'" if self.metrics is None else f"%.2f" % self.metrics["Mean Absolute Error (MAE)"],
                   "\'N/A\'" if self.metrics is None else f"%.2f" % self.metrics["Root Mean Squared Error (RMSE)"],
                   "\'N/A\'" if self.metrics is None else f"%.2f" % self.metrics["R-Squared (R2)"])
               )
        ax.set_title(ttl)
        return fig

    def plot_it(self):
        fig = self.__plot_it(self.year_range)

        pth = f"{self.rootpath}/predicting/{self.lr_range}_years_range"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"{pth}/Forest_Health_Classification_{self.year_range}({self.clf_name}).png"
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        # plt.close(fig)
        return fig

    @staticmethod
    def print_run_time(start_time):
        hours, remainder = divmod((datetime.now() - start_time).seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"run time: {hours:02}:{minutes:02}:{seconds:02}")
        return


if __name__ == '__main__':
    pass
