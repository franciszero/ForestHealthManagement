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
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, Normalize


class Foo:
    def __init__(self):
        self.rootpath = './h12v04'
        self.hdf_path = f'{self.rootpath}/data'
        self.hdf_files = sorted([f for f in os.listdir(self.hdf_path) if f.endswith('.hdf')],
                                key=self.__extract_date_from_filename)
        self.x1, self.x2 = 0, 4800
        self.y1, self.y2 = 0, 4800
        self.NDVI = np.zeros((len(self.hdf_files), self.x2 - self.x1, self.y2 - self.y1))
        self.VI_name = "NDVI"
        self.dates = []
        self.__foo()

        self.r_values = None
        self.std_errs = None
        self.intercepts = None
        self.pixel_classes = None
        self.p_values = None
        self.t_stats = None
        self.slopes = None
        self.sg_values = None
        self.years = None  # np.load('sg_years.npy')
        self.sg_values = None  # np.load('sg_values.npy')

        self.p_val_nan = 10
        pass

    @staticmethod
    def __extract_date_from_filename(fn):
        y = int(fn[9:13])
        doy = int(fn[13:16])  # day of year
        date = datetime(y, 1, 1) + timedelta(doy - 1)
        return date.strftime('%Y-%m-%d')

    def __foo(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        idx = 0
        for hdf_file in self.hdf_files:
            dt = self.__extract_date_from_filename(hdf_file)
            dt_obj = datetime.strptime(dt, '%Y-%m-%d')
            if not dt_obj.year <= 2010:
                continue
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
                if not os.path.exists(fp) or self.VI_name in sds:
                    print("\t%-40s" % sds, end=" ")
                    arr = ds[self.x1:self.x2 + 1, self.y1:self.y2 + 1]

                    # plot map
                    if os.path.exists(fp):
                        print(f"skipping plot {fp}")
                    else:
                        print(f"plotting map {fp}")
                    if self.VI_name in sds:
                        self.__plot_map(arr, dt, sds, fp)

                    # cache map
                    self.NDVI[idx, :, :] = arr
            idx += 1
            hdf.end()
        self.NDVI = self.NDVI[:len(self.dates), :, :]
        plt.close(fig)

        # fig, ax = plt.subplots(figsize=(16, 4))
        # for x in range(0, 4800, 500):
        #     for y in range(0, 4800, 500):
        #         foo.__plot_time_series(ax, x, y)

    @staticmethod
    def __plot_map(arr, dt, sds, fp):
        plt.imshow(arr, cmap='RdYlGn')
        plt.colorbar(label=sds)
        plt.title(f"{sds}_{dt}")
        plt.savefig(fp)
        plt.clf()

    def __plot_time_series(self, ax, x, y):
        df = pd.DataFrame(index=range(len(self.dates)), columns=["dt", "ndvi", "evi"])
        df["dt"] = self.dates
        df['dt'] = pd.to_datetime(df['dt'])
        df["ndvi"] = self.NDVI[:, x, y]
        df = pd.melt(df, id_vars=['dt'], value_vars=['ndvi'], var_name='hue', value_name='val')

        sns.lineplot(data=df, x='dt', y='val', hue='hue')
        # Setting grid, major ticks every year, minor ticks every month
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.grid(True, which='major', linestyle='-')
        ax.xaxis.grid(True, which='minor', linestyle='--')
        ax.yaxis.grid(True)

        pth = f"{self.rootpath}/time_series"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"%s/%4d_%4d.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()

    def compute_sg(self):
        year_sg = defaultdict(lambda: np.zeros((self.NDVI.shape[1], self.NDVI.shape[2])))
        for idx, date in enumerate([datetime.strptime(s, '%Y-%m-%d') for s in self.dates]):
            if 4 <= date.month <= 6:
                ndvi_without_water = np.where(self.NDVI[idx, :, :] == -3000, 0, self.NDVI[idx, :, :])
                year_sg[date.year] += ndvi_without_water
        self.years = np.array(sorted(list(year_sg.keys())))
        self.sg_values = np.array([year_sg[year] for year in self.years])
        return

    def linear_regress(self, start_idx, end_idx, start_year, end_year):
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

    def classify_pixels(self, end_idx, start_year, end_year):
        year_range = "%s~%s" % (start_year, end_year)

        is_forest_growth = self.slopes[:, :] >= 0
        is_forest_decline = self.slopes[:, :] < 0

        # mark water region
        p_val = np.where(self.sg_values[end_idx, :, :] == 0, self.p_val_nan, self.p_values)

        self.pixel_classes = np.zeros_like(p_val, dtype=np.int8)

        self.pixel_classes[is_forest_growth & (p_val > 0.0001) & (p_val <= 0.001)] = -3
        self.pixel_classes[is_forest_decline & (p_val <= 0.001)] = 3

        self.pixel_classes[is_forest_growth & (p_val > 0.001) & (p_val <= 0.01)] = -2
        self.pixel_classes[is_forest_decline & (p_val > 0.001) & (p_val <= 0.01)] = 2

        self.pixel_classes[is_forest_growth & (p_val > 0.01) & (p_val <= 0.05)] = -1
        self.pixel_classes[is_forest_decline & (p_val > 0.01) & (p_val <= 0.05)] = 1

        self.pixel_classes[p_val == self.p_val_nan] = -4

        self.pixel_classes[(p_val <= 1) & (p_val > 0.05)] = 0
        np.save(f'sg_clf_training_data_{end_year}.npy', self.pixel_classes)

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
        ax.set_title('Forest Health Classification based on SG trend %s' % year_range)

        pth = f"{self.rootpath}/clf"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"%s/yearly_SG_trend_%s.png" % (pth, year_range)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.close(fig)
        return

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

        df = pd.DataFrame(index=range(len(self.years)), columns=["dt", "ndvi"])
        df["dt"] = self.years
        df["ndvi"] = self.sg_values[:, x, y]

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

        fp = f"%s/%4d_%4d_sg_ndvi.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()
        pass


if __name__ == "__main__":

    foo = Foo()
    foo.compute_sg()  # save sg values

    pe = 3
    for i, year in enumerate(foo.years):
        if i + pe - 1 >= len(foo.years):
            break
        else:
            print(year, "~", year + pe - 1)
            foo.linear_regress(i, i + pe - 1, year, year + pe - 1)
            foo.classify_pixels(i + pe - 1, year, year + pe - 1)  # save sg clf training data

    # ana = SGAna(foo)
    # ana.analyze()

    # analysis.compute_sg()
    # plt.subplots(figsize=(12, 4))
    # for x1 in range(0, 4800, 500):
    #     for y1 in range(0, 4800, 500):
    #         analysis.plot_time_series_with_regression(x1, y1)
