from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import pandas as pd
from scipy.stats import linregress
from collections import defaultdict
from scipy.stats import t
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import UnivariateSpline
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
import h5py
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


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
        self.__foo()

        # for x in range(0, 4800, 500):
        #     for y in range(0, 4800, 500):
        #         self.plot_time_series(x, y, period=decomp_period)

        self.__compute_sg_values()  # save sg values
        # fig0, ax0 = plt.subplots(figsize=(12, 4))
        # for x1 in range(0, 4800, 500):
        #     for y1 in range(0, 4800, 500):
        #         self.plot_time_series_with_regression(x1, y1)
        # plt.close(fig0)

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
        #
        # start_year_idx, period = 2, 11
        # yrs = self.years[start_year_idx:start_year_idx + period]
        # year_range = "%s~%s" % (yrs[0], yrs[-1])
        # #
        # X_train, y_train, y_train_orig = self.get_Xy(start_year_idx, period, 400, 1800, 1000, 3800)
        # # self.__plot_it(y_train_orig, year_range)
        # X_test, y_test, y_test_orig = self.get_Xy(start_year_idx, period, 2000, 3000, 2000, 3000)
        # # self.__plot_it(y_test_orig, year_range)
        #
        # #
        # #
        # #
        # # i, period, h1, h2, w1, w2 = start_year_idx, period, 400, 1800, 1000, 3800
        # # i, period, h1, h2, w1, w2 = start_year_idx, period, 2000, 3000, 2000, 3000
        # # y = np.load(f'{self.rootpath}/data_training/sg_clf_training_data_{self.years[i]}~{self.years[i + period - 1]}.npy')
        # # y = y[h1:h2, w1:w2]
        # # y[y >= 2] = 2
        # # y[(y >= -3) & (y <= 1)] = 1
        # # y[y == -4] = 0
        # # unique, counts = np.unique(y.flatten(), return_counts=True)
        # # print(dict(zip(unique, counts)))
        # # __plot_it(y, year_range)
        # #
        # #
        # #
        #
        # # balance class weights
        # y_integers = np.argmax(y_train, axis=1)
        # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        # class_weight_dict = dict(enumerate(class_weights))
        #
        # # 构建模型
        # model = Sequential()
        # model.add(LSTM(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
        # model.add(LSTM(30, return_sequences=False))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(y_train.shape[1], activation='softmax'))  # y_train.shape[1] 应该是类别的数量
        #
        # # 编译模型
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #
        # # 添加早停以防过拟合
        # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        #
        # # 训练模型，使用类别权重和早停
        # history = model.fit(
        #     X_train,
        #     y_train,
        #     validation_split=0.1,
        #     epochs=50,
        #     batch_size=512,
        #     class_weight=class_weight_dict,
        #     callbacks=[early_stopping]
        # )
        #
        # # 评估模型
        # loss, accuracy = model.evaluate(X_test, y_test)
        # print(f"Test Loss: {loss}")
        # print(f"Test Accuracy: {accuracy}")
        # #
        # y_pred = model.predict(X_test)
        # y_pred_classes = np.argmax(y_pred, axis=1)
        # y_test_classes = np.argmax(y_test, axis=1)
        # cm = confusion_matrix(y_test_classes, y_pred_classes)
        # print(cm)
        # #
        # print(classification_report(y_test_classes, y_pred_classes))

    def lr_pred(self):
        X_train = self.sg_values[:-2].reshape(15, -1).T
        y_train = self.sg_values[-2].flatten()
        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = self.sg_values[1:-1].reshape(15, -1).T
        y_test = self.sg_values[-1].flatten()
        y_pred = model.predict(X_test)

        predicted_image = y_pred.reshape(self.sg_values.shape[1], self.sg_values.shape[2])
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE: {mae}")

    def get_Xy(self, i, period, h1, h2, w1, w2):
        X = self.sg_values[i: i + period, h1:h2, w1:w2].astype(np.int32)
        X = X.reshape((period, -1)).T

        y = np.load(
            f'{self.rootpath}/data_training/sg_clf_training_data_{self.years[i]}~{self.years[i + period - 1]}.npy')
        y = y[h1:h2, w1:w2]
        y[y >= 2] = 2
        y[(y >= -3) & (y <= 1)] = 1
        y[y == -4] = 0
        unique, counts = np.unique(y.flatten(), return_counts=True)
        print("y label count:", dict(zip(unique, counts)))
        y1 = y.reshape(-1)
        y1 = to_categorical(y1, num_classes=np.unique(y1).size)

        print("X shape:", X.shape)
        print("y shape:", y.shape)
        return X, y1, y

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
            self.print_run_time(t1)  # run time: 00:01:34
        else:
            print(f"Read NDVI data from {self.rootpath}")
            t1 = datetime.now()
            self.NDVI = np.zeros((len(self.hdf_files), self.x2 - self.x1, self.y2 - self.y1))
            self.__read_from_SDS()
            self.print_run_time(t1)  # run time: 00:04:25

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

            hdf = SD(os.path.join(self.hdf_path, hdf_file), SDC.READ)
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
                    if os.path.exists(fp): print(f"skipping plot {fp}")
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
