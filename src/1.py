from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns


class Foo:
    def __init__(self):
        self.rootpath = './h12v04'
        self.hdf_path = f'{self.rootpath}/data'
        self.hdf_files = sorted([f for f in os.listdir(self.hdf_path) if f.endswith('.hdf')],
                                key=self.__extract_date_from_filename)
        self.ndvi = np.zeros((len(self.hdf_files), 4800, 4800))
        self.evi = np.zeros((len(self.hdf_files), 4800, 4800))
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
        for idx, hdf_file in enumerate(self.hdf_files):
            dt = self.__extract_date_from_filename(hdf_file)
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
                if not os.path.exists(fp) or "NDVI" in sds or "EVI" in sds:
                    print("\t%-40s" % sds, end=" ")
                    arr = ds[:]

                    # plot map
                    if os.path.exists(fp):
                        print(f"skipping plot {fp}")
                    else:
                        print(f"plotting map {fp}")
                        self.__plot_map(arr, sds, fp)

                    # cache map
                    if "NDVI" in sds:
                        self.ndvi[idx] = arr
                    elif "EVI" in sds:
                        self.evi[idx] = arr
            hdf.end()

    @staticmethod
    def __plot_map(arr, sds, fp):
        plt.imshow(arr, cmap='RdYlGn')
        plt.colorbar(label=sds)
        plt.title(sds)
        plt.savefig(fp)
        plt.clf()

    def plot_time_series(self, x, y):
        pth = f"{self.rootpath}/time_series"
        if not os.path.exists(pth):
            os.makedirs(pth)

        df = pd.DataFrame(index=range(len(self.dates)), columns=["dt", "ndvi", "evi"])
        df["dt"] = self.dates
        df['dt'] = pd.to_datetime(df['dt'])
        df["ndvi"] = self.ndvi[:, x, y]
        df["evi"] = self.evi[:, x, y]
        df = pd.melt(df, id_vars=['dt'], value_vars=['ndvi', 'evi'], var_name='hue', value_name='val')

        sns.lineplot(data=df, x='dt', y='val', hue='hue')
        fp = f"%s/%4d_%4d.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()


if __name__ == "__main__":
    foo = Foo()

    plt.figure(figsize=(12, 4))
    for x1 in range(0, 4800, 500):
        for y1 in range(0, 4800, 500):
            foo.plot_time_series(x1, y1)
