from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import linregress, ttest_1samp
from collections import defaultdict
from scipy.stats import t
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import YearLocator, MonthLocator


class Foo:
    def __init__(self):
        self.rootpath = './h12v04'
        self.hdf_path = f'{self.rootpath}/data'
        self.hdf_files = sorted([f for f in os.listdir(self.hdf_path) if f.endswith('.hdf')],
                                key=self.__extract_date_from_filename)
        self.x1, self.x2 = 0, 4800
        self.y1, self.y2 = 0, 4800
        self.VIs = np.zeros((1, len(self.hdf_files), self.x2 - self.x1, self.y2 - self.y1))
        self.iNDVI = 0
        # self.iEVI = 1
        self.dates = []
        self.__foo()
        pass

    @staticmethod
    def __extract_date_from_filename(fn):
        year = int(fn[9:13])
        doy = int(fn[13:16])  # day of year
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        return date.strftime('%Y-%m-%d')

    def __foo(self):
        plt.figure(figsize=(10, 10))
        idx = 0
        for hdf_file in self.hdf_files:
            dt = self.__extract_date_from_filename(hdf_file)
            dt_obj = datetime.strptime(dt, '%Y-%m-%d')
            # if dt_obj.year >= 2019:  # dt_obj.month > 6 or dt_obj.month < 4:
            #     continue
            print(dt)
            self.dates.append(dt)

            hdf = SD(os.path.join(self.hdf_path, hdf_file), SDC.READ)
            for _, sds in enumerate(hdf.datasets().keys()):
                # get dataset
                ds = hdf.select(sds)

                # check folder
                sds = sds.replace(' ', '_')
                pth = f"{self.rootpath}/{sds}"
                if not os.path.exists(pth):
                    os.makedirs(pth)

                # check file
                fp = f"{pth}/{dt}.png"
                if not os.path.exists(fp) or "NDVI" in sds:  # or "EVI" in sds:
                    print("\t%-40s" % sds, end=" ")
                    arr = ds[self.x1:self.x2 + 1, self.y1:self.y2 + 1]

                    # plot map
                    if os.path.exists(fp):
                        print(f"skipping plot {fp}")
                    else:
                        print(f"plotting map {fp}")
                        self.__plot_map(arr, sds, fp)

                    # cache map
                    if "NDVI" in sds:
                        self.VIs[self.iNDVI, idx, :, :] = arr
                    # elif "EVI" in sds:
                    #     self.VIs[self.iEVI, idx, :, :] = arr
            idx += 1
            hdf.end()

    @staticmethod
    def __plot_map(arr, sds, fp):
        plt.imshow(arr, cmap='RdYlGn')
        plt.colorbar(label=sds)
        plt.title(sds)
        plt.savefig(fp)
        plt.clf()

    def plot_time_series(self, ax, x, y):
        pth = f"{self.rootpath}/time_series"
        if not os.path.exists(pth):
            os.makedirs(pth)

        df = pd.DataFrame(index=range(len(self.dates)), columns=["dt", "ndvi", "evi"])
        df["dt"] = self.dates
        df['dt'] = pd.to_datetime(df['dt'])
        df["ndvi"] = self.VIs[self.iNDVI, :, x, y]
        # df["evi"] = self.VIs[self.iEVI, :, x, y]
        df = pd.melt(df, id_vars=['dt'], value_vars=['ndvi', 'evi'], var_name='hue', value_name='val')

        sns.lineplot(data=df, x='dt', y='val', hue='hue')
        # Setting grid, major ticks every year, minor ticks every month
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.grid(True, which='major', linestyle='-')
        ax.xaxis.grid(True, which='minor', linestyle='--')
        ax.yaxis.grid(True)

        fp = f"%s/%4d_%4d.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()


class SGAna:
    def __init__(self, root, dates, ndvi_data):
        self.pixel_classes = None
        self.p_values = None
        self.t_stats = None
        self.slopes = None
        self.sg_values = None
        self.rootpath = root
        self.dates = [datetime.strptime(s, '%Y-%m-%d') for s in dates]
        self.ndvi = ndvi_data[0, :len(self.dates), :, :]

        self.map_shape = (self.ndvi.shape[1], self.ndvi.shape[2])
        self.years = None
        self.p_val_nan = 10

    def compute_sg(self):
        year_sg = defaultdict(lambda: np.zeros(self.map_shape))
        for i, date in enumerate(self.dates):
            if 4 <= date.month <= 6:
                ndvi_without_water = np.where(self.ndvi[i, :, :] == -3000, 0, self.ndvi[i, :, :])
                year_sg[date.year] += ndvi_without_water
        self.years = sorted(list(year_sg.keys()))
        self.sg_values = np.array([year_sg[year] for year in self.years])
        return

    def perform_regression(self):
        years_col = self.years[:, np.newaxis, np.newaxis]
        x_mean = np.mean(years_col)
        y_mean = np.mean(self.sg_values, axis=0)
        num = np.sum((years_col - x_mean) * (self.sg_values - y_mean), axis=0)
        den = np.sum((years_col - x_mean) ** 2)
        slopes = num / den
        # x, y = 0, -1
        # plt.clf()
        # sns.lineplot(x=self.years, y=sg[:, x, y])
        # sns.lineplot(x=self.years, y=[slopes[x, y] * i for i in range(len(self.years))])
        # print(slopes[x, y])
        self.slopes, _ = slopes, None
        return

    def significance_test(self):
        n = len(self.years)

        # 计算y的均值
        y_mean = np.mean(self.sg_values, axis=0)

        # 将years转化为3D，以便于矩阵计算
        years_col = self.years[:, np.newaxis, np.newaxis]

        # 计算每个像素的残差平方和
        sse = np.sum((self.sg_values - (self.slopes * years_col + y_mean)) ** 2, axis=0)

        # 计算每个像素的标准误差
        se = np.sqrt(sse / (n - 2)) / np.sqrt(np.sum((years_col - np.mean(self.years)) ** 2))

        # 计算t统计量
        t_statistic = self.slopes / se

        # 计算p-value（双尾）
        df = n - 2
        p_values = t.sf(np.abs(t_statistic), df) * 2

        self.t_stats, self.p_values = t_statistic, p_values
        return

    def significance_test1(self, sg_values, slopes):
        # Compute the t-statistics for each pixel to test the significance of its slope against 0
        t_stats = np.zeros_like(sg_values[0])
        p_values = np.zeros_like(sg_values[0])

        for i in range(sg_values.shape[1]):
            for j in range(sg_values.shape[2]):
                residuals = sg_values[:, i, j] - (slopes[i, j] * self.years)
                t_stat, p_val = ttest_1samp(residuals, 0)
                t_stats[i, j] = t_stat
                p_values[i, j] = p_val

        return t_stats, p_values

    def significance_test_matrix(self, sg_values, slopes):
        n = sg_values.shape[0]
        residuals = sg_values - (slopes[np.newaxis, :, :] * self.years[:, np.newaxis, np.newaxis])
        t_stats = residuals.mean(axis=0) / (residuals.std(axis=0, ddof=1) / np.sqrt(n))
        p_values = 2 * t.sf(np.abs(t_stats), n - 1)
        return t_stats, p_values

    def classify_pixels(self):
        self.pixel_classes = np.zeros_like(self.p_values, dtype=int)
        self.pixel_classes[self.p_values <= 0.001] = 3
        self.pixel_classes[(self.p_values > 0.001) & (self.p_values <= 0.01)] = 2
        self.pixel_classes[(self.p_values > 0.01) & (self.p_values <= 0.05)] = 1
        self.pixel_classes[(self.p_values <= 1) & (self.p_values > 0.05)] = 0
        self.pixel_classes[self.p_values == self.p_val_nan] = -1
        return

    def analyze(self):
        self.compute_sg()
        self.perform_regression()
        self.significance_test()
        self.classify_pixels()

        cmap = plt.get_cmap('RdYlGn_r')
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(self.pixel_classes, cmap=cmap)
        cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3], orientation='vertical')
        cbar.ax.set_yticklabels(['Healthy', 'Uncertain', 'Declining', 'Severely Declining'])
        ax.set_title('Forest Health Classification based on SG trend')
        plt.show()

    def plot_classification(self):
        cmap = plt.get_cmap('RdYlGn_r')
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(self.pixel_classes, cmap=cmap)
        cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3], orientation='vertical')
        cbar.ax.set_yticklabels(['Healthy', 'Uncertain', 'Declining', 'Severely Declining'])
        ax.set_title('Forest Health Classification based on SG trend')
        plt.show()

        pth = f"{self.rootpath}/clf"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"%s/%s.png" % (pth, max(self.dates).strftime('%Y-%m-%d'))
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()

    def plot_time_series_with_regression(self, x, y):
        pth = f"{self.rootpath}/time_series"
        if not os.path.exists(pth):
            os.makedirs(pth)

        df = pd.DataFrame(index=range(len(analysis.years)), columns=["dt", "ndvi"])
        df["dt"] = analysis.years
        df["ndvi"] = analysis.sg_values[:, x, y]

        slope, intercept, r_value, p_value, std_err = linregress(df["dt"].astype(int), df["ndvi"])

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

        fp = f"%s/%4d_%4d_sg_ndvi.png" % (pth, x1, y1)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()
        pass


if __name__ == "__main__":
    foo = Foo()

    # fig, ax0 = plt.subplots(figsize=(12, 4))
    # for x1 in range(0, 4800, 500):
    #     for y1 in range(0, 4800, 500):
    #         foo.plot_time_series(ax0, x1, y1)

    # Use the class
    analysis = SGAna(foo.rootpath, foo.dates, foo.VIs)
    analysis.analyze()

    analysis.compute_sg()

    plt.subplots(figsize=(12, 4))
    for x1 in range(0, 4800, 500):
        for y1 in range(0, 4800, 500):
            analysis.plot_time_series_with_regression(x1, y1)

    # analysis.analyze()
    # analysis.plot_classification()
