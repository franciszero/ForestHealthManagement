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
from src.ModelSelection import ModelSelection


class Foo:
    def __init__(self):
        self.rootpath = '../h12v04'
        self.hdf_path = f'{self.rootpath}/data'
        self.hdf_files = sorted([f for f in os.listdir(self.hdf_path) if f.endswith('.hdf')],
                                key=self.__extract_date_from_filename)
        self.x1, self.x2 = 0, 4800
        self.y1, self.y2 = 0, 4800
        self.NDVI = None

        self.VI_name = "NDVI"
        self.dates = []
        self.years = None
        for hdf_file in self.hdf_files:
            y, dt = self.__extract_date_from_filename(hdf_file)
            # print(dt)
            self.dates.append(dt)

        self.r_values = None
        self.std_errs = None
        self.intercepts = None
        self.pixel_classes = None
        self.p_values = None
        self.t_stats = None
        self.slopes = None
        self.sg_values = None

        self.p_val_nan = 10
        pass

    def process(self, decomp_period=24):
        # self.__foo()

        # for x in range(0, 4800, 500):
        #     for y in range(0, 4800, 500):
        #         self.plot_time_series(x, y, period=decomp_period)

        self.__compute_sg_values()  # save sg values
        # fig0, ax0 = plt.subplots(figsize=(12, 4))
        # for x1 in range(0, 4800, 500):
        #     for y1 in range(0, 4800, 500):
        #         self.plot_time_series_with_regression(x1, y1)
        # plt.close(fig0)

        ms = ModelSelection(self.sg_values)  # test_xy=(0, 0), test_hw=(4800, 4800)

        ms.train_nn(1)
        ms.train_nn(2)
        ms.train_nn(3)
        ms.train_nn(4)

        name_lr, model_lr, y_pred_lr, metrics_lr = ms.train_lr()
        name_xgb, model_xgb, y_pred_xgb, metrics_xgb = ms.train_xgb()
        # name_rf, model_rf, y_pred_rf, metrics_rf = ms.train_rf()
        name_dt, model_dt, y_pred_dt, metrics_dt = ms.train_dt()
        # name_svr, model_svr, y_pred_svr, metrics_svr = ms.train_svr()
        name_en, model_en, y_pred_en, metrics_en = ms.train_ensemble_model([(name_lr, model_lr),
                                                                            (name_xgb, model_xgb),
                                                                            # (name_rf, model_rf),
                                                                            (name_dt, model_dt),
                                                                            # (name_svr, model_svr)
                                                                            ])

        '''
        把计算线性回归的部分，和像素分类的部分，写到一个新的 class 里
        可视化对比 map fact 森林健康染色图，和预测森林健康染色图
        做 confuse matrix 的 heatmap，看每个模型在森林健康/不健康上的预测准确率
        '''

        # for lr_period in range(7, 8):  # range(3, 14):
        #     for i, year in enumerate(self.years):
        #         if i + lr_period - 1 >= len(self.years):
        #             break
        #         else:
        #             start_year, end_year = year, year + lr_period - 1
        #             end_idx = i + lr_period - 1
        #
        #             print(f"Forest Health Classification with SG trend from {year} to {year + lr_period - 1}")
        #             pth = f"{self.rootpath}/data_training"
        #             year_range = "%s~%s" % (start_year, end_year)
        #             if not os.path.exists(f'{pth}/sg_clf_training_data_{year_range}.npy'):
        #                 self.linear_regress(i, i + lr_period - 1)
        #                 year_range = self.classify_pixels(end_idx, start_year, end_year)
        #                 self.plot_it(year_range, lr_period)
        #             else:
        #                 self.pixel_classes = np.load(f'{pth}/sg_clf_training_data_{year_range}.npy')


    # def nn1(self):
    #     time_steps = 14
    #     h, w = 1000, 1000
    #     channels = 1
    #     weight_file = 'best_spatiotemporal_model.h5'
    #
    #     h1, w1 = 0, 0
    #     h2, w2 = h1 + h, w1 + w
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(self.sg_values[0:14, h1:h2, w1:w2].reshape(time_steps, -1).T)
    #     X_train = X_train.reshape(-1, time_steps, h, w, channels)
    #     y_train = self.sg_values[14, h1:h2, w1:w2].reshape(1, -1)
    #
    #     h1, w1 = 1000, 1000
    #     h2, w2 = h1 + h, w1 + w
    #     X_val = scaler.transform(self.sg_values[1:15, h1:h2, w1:w2].reshape(time_steps, -1).T)
    #     X_val = X_val.reshape(-1, time_steps, h, w, channels)
    #     y_val = self.sg_values[15, h1:h2, w1:w2].reshape(1, -1)
    #
    #     h1, w1 = 2000, 2000
    #     h2, w2 = h1 + h, w1 + w
    #     X_test = scaler.transform(self.sg_values[2:16, h1:h2, w1:w2].reshape(time_steps, -1).T)
    #     X_test = X_test.reshape(-1, time_steps, h, w, channels)
    #     y_test = self.sg_values[16, h1:h2, w1:w2].reshape(1, -1)
    #
    #     print("X_train.shape = ", X_train.shape)
    #     print("y_train.shape = ", y_train.shape)
    #     print("X_val.shape = ", X_val.shape)
    #     print("y_val.shape = ", y_val.shape)
    #     print("X_test.shape = ", X_test.shape)
    #     print("y_test.shape = ", y_test.shape)
    #
    #     # # balance
    #     # plt.figure(figsize=(12, 6))
    #     # plt.subplot(1, 2, 1)
    #     # plt.hist(y_train.flatten(), bins=50, color='blue', alpha=0.7)
    #     # plt.title('Train Data Distribution')
    #     # plt.subplot(1, 2, 2)
    #     # plt.hist(y_test.flatten(), bins=50, color='green', alpha=0.7)
    #     # plt.title('Test Data Distribution')
    #     # plt.show()
    #
    #     '''
    #     model = Sequential()
    #     model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'), input_shape=(time_steps, height, width, channels)))
    #     model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #     model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu')))
    #     model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #     model.add(TimeDistributed(Flatten()))
    #     model.add(LSTM(units=32, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(units=50, activation='relu'))
    #     model.add(Dense(units=height * width, activation='linear'))
    #     opt = Adam(learning_rate=0.001)
    #     model.compile(optimizer=opt, loss='mean_squared_error')
    #     model.summary()
    #     '''
    #     '''
    #     # 定义模型
    #     model = Sequential()
    #     model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)), input_shape=(time_steps, height, width, channels)))
    #     model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #     model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001))))
    #     model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #     model.add(TimeDistributed(GlobalAveragePooling2D()))
    #     # model.add(TimeDistributed(Flatten()))
    #     model.add(LSTM(units=16, activation='relu', kernel_regularizer=l2(0.001)))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.001)))
    #     model.add(Dense(units=height * width, activation='linear'))
    #     optimizer = Adam()
    #     model.compile(optimizer=optimizer, loss='mean_squared_error')
    #     model.summary()
    #     # ----------------------------
    #     # Mean Absolute Error (MAE): 7246.1902
    #     # Mean Squared Error (MSE): 167638818.1418
    #     # Root Mean Squared Error (RMSE): 7246.1902
    #     # R-Squared (R2): nan
    #     # ----------------------------
    #     '''
    #     model = Sequential()
    #     # spacial
    #     model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), kernel_regularizer=l2(0.001)), input_shape=(time_steps, h, w, channels)))
    #     model.add(TimeDistributed(BatchNormalization()))
    #     model.add(TimeDistributed(Activation('relu')))
    #     model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #
    #     model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), kernel_regularizer=l2(0.001))))
    #     model.add(TimeDistributed(BatchNormalization()))
    #     model.add(TimeDistributed(Activation('relu')))
    #     model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #
    #     model.add(TimeDistributed(Flatten()))
    #
    #     # temporal
    #     model.add(LSTM(units=20, activation='relu', kernel_regularizer=l2(0.001), return_sequences=False))
    #     model.add(Dropout(0.5))
    #     # model.add(BatchNormalization())
    #     # output
    #     model.add(Dense(units=h * w, activation='linear'))
    #     # optimizer
    #     model.compile(optimizer=Adam(), loss='mean_squared_error')
    #     model.summary()
    #
    #     # early stop & check point
    #     callbacks = [
    #         EarlyStopping(monitor='val_loss', patience=15),
    #         ModelCheckpoint(weight_file, save_best_only=True)
    #     ]
    #
    #     # training
    #     history = model.fit(
    #         X_train, y_train,
    #         epochs=300,
    #         batch_size=1024,
    #         validation_data=(X_val, y_val),
    #         verbose=2,
    #         callbacks=callbacks
    #     )
    #
    #     # load weights
    #     model.load_weights(weight_file)
    #     loss = model.evaluate(X_test, y_test)
    #     print(f"Test Loss: {loss}")
    #
    #     '''
    #     # performance
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(history.history['loss'], label='Train Loss')
    #     plt.plot(history.history['val_loss'], label='Validation Loss')
    #     plt.title('Model Performance')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend()
    #     plt.show()
    #     '''
    #
    #     # Predict and evaluate
    #     y_pred = model.predict(X_test)
    #     mae = mean_absolute_error(y_test, y_pred)
    #     mse = mean_squared_error(y_test, y_pred)
    #     rmse = mean_squared_error(y_test, y_pred, squared=False)
    #     r2 = r2_score(y_test, y_pred)
    #
    #     # Print evaluation metrics
    #     print("LSTM Model Evaluation Metrics:")
    #     print("----------------------------")
    #     print(f"Mean Absolute Error (MAE): {mae:.4f}")
    #     print(f"Mean Squared Error (MSE): {mse:.4f}")
    #     print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    #     print(f"R-Squared (R2): {r2:.4f}")
    #     print("----------------------------")

    def model_training(self, model, time_steps=14, h1=0, w1=0, h=1000, w=1000):
        h2, w2 = h1 + h, w1 + w
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.sg_values[0:14, h1:h2, w1:w2].reshape(time_steps, -1).T)
        y_train = self.sg_values[14, h1:h2, w1:w2].flatten()
        model.fit(X_train, y_train)
        return scaler, model

    def model_val(self, y_pred, y_test):
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # Formatting and printing the results
        metrics = {
            'Mean Absolute Error (MAE)': mae,
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'R-Squared (R2)': r2
        }

        print("Model Evaluation Metrics:")
        print("----------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("----------------------------")

        return metrics

    @staticmethod
    def __extract_date_from_filename(fn):
        y = int(fn[9:13])
        doy = int(fn[13:16])  # day of year
        date = datetime(y, 1, 1) + timedelta(doy - 1)
        return y, date.strftime('%Y-%m-%d')

    def __foo(self):
        ndvi_h5 = f'./ndvi.h5'
        if os.path.exists(ndvi_h5):
            print(f"Read NDVI data from {ndvi_h5}")
            t1 = datetime.now()
            with h5py.File(ndvi_h5, 'r') as hf:
                self.NDVI = hf['NDVI'][:]
            self.print_run_time(t1)  # run time: 00:01:34 / 00:00:16
        else:
            print(f"Read NDVI data from {self.rootpath}")
            t1 = datetime.now()
            self.NDVI = np.zeros((len(self.hdf_files), self.x2 - self.x1, self.y2 - self.y1))
            self.__read_from_SDS()
            self.print_run_time(t1)  # run time: 00:04:25 / 00:02:14

            print(f"save NDVI data into {ndvi_h5}")
            t1 = datetime.now()
            with h5py.File(ndvi_h5, 'w') as hf:
                hf.create_dataset('NDVI', data=self.NDVI)
            self.print_run_time(t1)  # run time: 00:05:14

    def __read_from_SDS(self):
        idx = 0
        for hdf_file, dt in zip(self.hdf_files, self.dates):
            y, dt = self.__extract_date_from_filename(hdf_file)
            print(dt)
            self.dates.append(dt)

            hdf_file_path = os.path.join(self.hdf_path, hdf_file)
            hdf = SD(hdf_file_path, SDC.READ)
            for _, sds in enumerate(hdf.datasets().keys()):
                # check folder
                sds1 = sds.replace(' ', '_')
                pth = f"{self.rootpath}/{sds1}"
                if not os.path.exists(pth):
                    os.makedirs(pth)

                # check file
                fp = f"{pth}/{dt}.png"
                if not os.path.exists(fp) or self.VI_name in sds1:
                    print("\t%-40s" % sds1, end=" ")
                    # get dataset
                    ds = hdf.select(sds)
                    arr = ds[self.x1:self.x2 + 1, self.y1:self.y2 + 1]

                    # plot map
                    if os.path.exists(fp):
                        print(f"skipping plot {fp}")
                    else:
                        print(f"plotting map {fp}")
                        # if self.VI_name in sds:
                        self.__plot_map(arr, dt, sds1, fp)

                    # cache map
                    self.NDVI[idx, :, :] = arr
            idx += 1
            hdf.end()
        self.NDVI = self.NDVI[:len(self.dates), :, :]

    @staticmethod
    def __plot_map(arr, dt, sds, fp):
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(arr, cmap='RdYlGn')
        major_ticks = np.arange(0, arr.shape[1] + 1, 1000)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        minor_ticks = np.arange(0, arr.shape[1] + 1, 200)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        ax.tick_params(axis='both', which='minor', size=0)

        ax.grid(True, which='both', color='darkblue', linewidth=0.5, linestyle='--')
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=0.8)

        plt.colorbar(im, label=sds)
        plt.title(f"{sds}_{dt}")
        plt.savefig(fp)
        plt.close(fig)

    @staticmethod
    def __set_grids(ax):
        # Setting grid, major ticks every year, minor ticks every month
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.tick_params(axis='both', which='minor', size=0)

        ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--')
        ax.grid(which='minor', alpha=0.4)
        ax.grid(which='major', alpha=0.8)

    def _get_ts(self, x, y):
        df = pd.DataFrame(index=range(len(self.dates)), columns=["dt", "ndvi"])
        df["dt"] = self.dates
        df['dt'] = pd.to_datetime(df['dt'])
        df["ndvi"] = self.NDVI[:, x, y]
        return df

    def _plot_ax1(self, ax, dt, ts):
        self.__set_grids(ax)
        ax.plot(dt, ts)
        ax.set_title('')
        ax.set_ylabel('NDVI Time Series')
        ax.set_ylim(-2000, 10000)

    def _plot_ax2(self, ax, dt, trend):
        self.__set_grids(ax)
        tr = pd.DataFrame(index=range(len(dt)), columns=["dt", "trend"])
        tr["dt"] = dt
        tr["trend"] = trend
        ax.plot(tr[~tr["trend"].isna()]["dt"], tr[~tr["trend"].isna()]["trend"])
        ax.plot(tr[tr["trend"].isna()]["dt"], tr[tr["trend"].isna()]["trend"].fillna(0), alpha=0)
        ax.set_title('')
        ax.set_ylabel('Trend')
        ax.set_ylim(-2000, 10000)

    def _plot_ax3(self, ax, dt, seasonal):
        self.__set_grids(ax)
        ax.plot(dt, seasonal)
        ax.set_title('')
        ax.set_ylabel('Seasonal')

    def _plot_ax4(self, ax, dt, resid):
        self.__set_grids(ax)

        markerline, stemline, baseline = ax.stem(dt, resid)
        plt.setp(markerline, 'markerfacecolor', 'blue', 'markersize', 1.2)
        plt.setp(stemline, 'linewidth', 0.2)
        plt.setp(baseline, 'linewidth', 0)

        dt_num = mdates.date2num(dt)
        smooth_x_num = np.linspace(dt_num.min(), dt_num.max(), 1000)
        spline = UnivariateSpline(dt_num, resid.fillna(0), s=0)
        smooth_y = spline(smooth_x_num)
        smooth_x = mdates.num2date(smooth_x_num)

        ax.fill_between(smooth_x, smooth_y, 0, facecolor='C0', alpha=0.4)
        ax.set_title('')
        ax.set_ylabel('Residual (Stem Plot)')
        ax.set_xlabel('Year')

    def plot_time_series(self, x, y, period=12 * 2):
        df = self._get_ts(x, y)
        # seasonal_decompose(df['ndvi'], model='additive', period=period)
        result = STL(df["ndvi"], period=period).fit()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 8))

        dt_range = "(%s ~ %s)" % (df["dt"].min().strftime("%Y-%m-%d"), df["dt"].max().strftime("%Y-%m-%d"))
        fig.suptitle(f'NDVI Time Series Seasonal Decomposition {dt_range} (x:{x} y:{y})')
        self._plot_ax1(ax1, df["dt"], df["ndvi"])
        self._plot_ax2(ax2, df["dt"], result.trend)
        self._plot_ax3(ax3, df["dt"], result.seasonal)
        self._plot_ax4(ax4, df["dt"], result.resid)
        fig.tight_layout()

        pth = f"{self.rootpath}/time_series"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"%s/%4d_%4d.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.close(fig)

    # def plot_time_series(self, ax, x, y):
    #     df = pd.DataFrame(index=range(len(self.dates)), columns=["dt", "ndvi", "evi"])
    #     df["dt"] = self.dates
    #     df['dt'] = pd.to_datetime(df['dt'])
    #     df["ndvi"] = self.NDVI[:, x, y]
    #     df = pd.melt(df, id_vars=['dt'], value_vars=['ndvi'], var_name='hue', value_name='val')
    #
    #     ax = sns.lineplot(data=df, x='dt', y='val', hue='hue')
    #     ax.set_title('NDVI Time Series')
    #
    #     result = seasonal_decompose(df['ndvi'], model='additive', period=120)
    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 8))
    #
    #     ax1.plot(df["dt"], df["ndvi"])
    #     # Setting grid, major ticks every year, minor ticks every month
    #     ax1.xaxis.set_major_locator(YearLocator())
    #     ax1.xaxis.set_minor_locator(MonthLocator())
    #     ax1.tick_params(axis='both', which='minor', size=0)
    #
    #     ax1.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--')
    #     ax1.grid(which='minor', alpha=0.4)
    #     ax1.grid(which='major', alpha=0.8)
    #
    #     ax1.plot(df["dt"], df["ndvi"])
    #     ax1.set_title('')
    #     ax1.set_ylabel('NDVI Time Series')
    #
    #     ax2.plot(df["dt"], result.trend)
    #     ax2.set_title('')
    #     ax2.set_ylabel('Trend')
    #
    #     ax3.plot(df["dt"], result.seasonal)
    #     ax3.set_title('')
    #     ax3.set_ylabel('Seasonal')
    #
    #     ax4.scatter(df["dt"], result.resid)
    #     ax4.set_title('')
    #     ax4.set_ylabel('Residual')
    #     ax4.set_xlabel('Year')
    #
    #     pth = f"{self.rootpath}/time_series"
    #     if not os.path.exists(pth):
    #         os.makedirs(pth)
    #     fp = f"%s/%4d_%4d.png" % (pth, x, y)
    #     plt.savefig(fp)
    #     print(f"plot saved at {fp}")
    #     plt.clf()

    def __compute_sg_values(self):
        sg_h5 = f'./sg.h5'
        if os.path.exists(sg_h5):
            print(f"Read SG values from {sg_h5}")
            t1 = datetime.now()
            with h5py.File(sg_h5, 'r') as hf:
                self.sg_values = hf['SG'][:]
            self.years = np.load('sg_trend_yearly_range.npy')
            self.print_run_time(t1)  # run time: 00:00:05
        else:
            print(f"Compute SG values")
            t1 = datetime.now()
            self.__compute_sg()
            self.print_run_time(t1)  # run time: 00:00:55

            print(f"Save SG values into {sg_h5}")
            t1 = datetime.now()
            with h5py.File(sg_h5, 'w') as hf:
                hf.create_dataset('SG', data=self.sg_values)
            self.print_run_time(t1)  # run time: 00:00:13

    def __compute_sg(self):
        year_sg = defaultdict(lambda: np.zeros((self.NDVI.shape[1], self.NDVI.shape[2])))
        for idx, date in enumerate([datetime.strptime(s, '%Y-%m-%d') for s in self.dates]):
            if 4 <= date.month <= 6:
                ndvi_without_water = np.where(self.NDVI[idx, :, :] == -3000, 0, self.NDVI[idx, :, :])
                year_sg[date.year] += ndvi_without_water
        self.years = np.array(sorted(list(year_sg.keys())))
        np.save(f'./sg_trend_yearly_range.npy', self.years)
        self.sg_values = np.array([year_sg[y] for y in self.years])
        print(f"Collected SG Values for {len(self.years)} years.")

    def linear_regress(self, start_idx, end_idx):
        yrs = self.years[start_idx: end_idx + 1]
        sgs = self.sg_values[start_idx:end_idx + 1, :, :]

        years_col = yrs[:, np.newaxis, np.newaxis]
        x_mean = np.mean(years_col)
        y_mean = np.mean(sgs, axis=0)

        num = np.sum((years_col - x_mean) * (sgs - y_mean), axis=0)
        den = np.sum((years_col - x_mean) ** 2)

        # Calculate slope and intercept
        self.slopes = num / den
        self.intercepts = y_mean - self.slopes * x_mean

        # Calculate covariance and variance for r_value calculation
        covariance = np.sum((yrs[:, np.newaxis, np.newaxis] - x_mean) * (sgs - y_mean), axis=0)
        variance_x = np.sum((yrs - x_mean.squeeze()) ** 2)
        variance_y = np.sum((sgs - y_mean) ** 2, axis=0)
        denominator = variance_x * variance_y
        self.r_values = np.where(variance_y != 0, covariance / np.sqrt(denominator), 0)

        # Calculate p_value and std_err
        n = len(self.years)
        t_stat = self.r_values * np.sqrt(n - 2) / np.sqrt(1 - self.r_values ** 2)

        # p value
        df = n - 2
        self.p_values = t.sf(np.abs(t_stat), df) * 2

        # std err
        self.std_errs = np.sqrt((1 - self.r_values ** 2) * np.var(self.sg_values, axis=0) / (n - 2))
        return

    @staticmethod
    def __plot_it(y, year_range):
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
        cax = ax.imshow(y, cmap=cmap_custom, norm=norm)

        major_ticks = np.arange(0, y.shape[1] + 1, 1000)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        minor_ticks = np.arange(0, y.shape[1] + 1, 200)
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
        ax.set_title('Forest Health Classification based on SG trend %s' % year_range)
        return fig

    def plot_it(self, period, year_range):
        fig = self.__plot_it(self.pixel_classes, year_range)

        pth = f"{self.rootpath}/clf/{period}_years"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"%s/yearly_SG_trend_%s.png" % (pth, year_range)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.close(fig)

    def classify_pixels(self, end_idx, start_year, end_year):
        is_forest_growth = self.slopes[:, :] >= 0
        is_forest_decline = self.slopes[:, :] < 0

        # mark water region
        p_val = np.where(self.sg_values[end_idx, :, :] == 0, self.p_val_nan, self.p_values)

        self.pixel_classes = np.zeros_like(p_val, dtype=np.int8)

        self.pixel_classes[is_forest_growth & (p_val <= 0.001)] = -3
        self.pixel_classes[is_forest_decline & (p_val <= 0.001)] = 3

        self.pixel_classes[is_forest_growth & (p_val > 0.001) & (p_val <= 0.01)] = -2
        self.pixel_classes[is_forest_decline & (p_val > 0.001) & (p_val <= 0.01)] = 2

        self.pixel_classes[is_forest_growth & (p_val > 0.01) & (p_val <= 0.05)] = -1
        self.pixel_classes[is_forest_decline & (p_val > 0.01) & (p_val <= 0.05)] = 1

        self.pixel_classes[p_val == self.p_val_nan] = -4

        self.pixel_classes[(p_val <= 1) & (p_val > 0.05)] = 0

        pth = f"{self.rootpath}/data_training"
        if not os.path.exists(pth):
            os.makedirs(pth)
        year_range = "%s~%s" % (start_year, end_year)
        np.save(f'{pth}/sg_clf_training_data_{year_range}.npy', self.pixel_classes)
        return year_range

    def plot_time_series_with_regression(self, x, y):
        pth = f"{self.rootpath}/time_series"
        if not os.path.exists(pth):
            os.makedirs(pth)

        df = pd.DataFrame(index=range(len(self.years)), columns=["dt", "ndvi"])
        df["dt"] = self.years.astype(int)
        df["ndvi"] = self.sg_values[:, x, y]

        slope, intercept, r_value, p_value, std_err = linregress(df["dt"].astype(int), df["ndvi"])

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.bar(df["dt"], df["ndvi"])
        plt.plot(df["dt"], slope * df["dt"].astype(int) + intercept, color='red')

        text = f"y = {slope:.2f}x + {intercept:.2f}"
        plt.annotate(text, (0.05, 0.95), xycoords='axes fraction', fontsize=12, color="red")
        plt.annotate(f"r_value = {r_value:.2f}", (0.05, 0.88), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"p_value = {p_value:.4f}", (0.05, 0.81), xycoords='axes fraction', fontsize=12)
        plt.annotate(f"std_err = {std_err:.2f}", (0.05, 0.74), xycoords='axes fraction', fontsize=12)

        plt.title("NDVI Time Series with Regression")
        plt.xlabel("Year")
        plt.ylabel("NDVI Value")

        fp = f"%s/%4d_%4d_sg_ndvi.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()
        pass

    @staticmethod
    def print_run_time(start_time):
        hours, remainder = divmod((datetime.now() - start_time).seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"run time: {hours:02}:{minutes:02}:{seconds:02}")
        return


if __name__ == "__main__":
    foo = Foo()
    foo.process(24)
