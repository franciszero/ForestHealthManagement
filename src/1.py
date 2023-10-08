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
        self.ndvi = []
        self.evi = []
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
        for hdf_file in self.hdf_files:
            dt = self.__extract_date_from_filename(hdf_file)
            print(dt)
            self.dates.append(dt)

            hdf = SD(os.path.join(self.hdf_path, hdf_file), SDC.READ)
            for idx, sds in enumerate(hdf.datasets().keys()):
                print("\t%-40s" % sds, end=" ")
                ds = hdf.select(sds)
                arr = ds[:]

                is_break = False
                if "NDVI" in sds:
                    self.ndvi.append(arr)
                elif "EVI" in sds:
                    self.evi.append(arr)
                else:
                    is_break = True

                sds = sds.replace(' ', '_')
                pth = f"{self.rootpath}/{sds}"
                if not os.path.exists(pth):
                    os.makedirs(pth)
                fp = f"{pth}/{dt}.png"
                if not os.path.exists(fp):
                    print(f"plotting map {fp}")
                    self.__plot_map(arr, sds, fp)
                else:
                    print(f"skipping plot {fp}")
                    if is_break:
                        break
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
        df["ndvi"] = np.array(self.ndvi)[:, x, y]
        df["evi"] = np.array(self.evi)[:, x, y]
        # 使用 pd.melt 重塑 DataFrame
        df = pd.melt(df, id_vars=['dt'], value_vars=['ndvi', 'evi'], var_name='hue', value_name='val')
        sns.lineplot(data=df, x='dt', y='val', hue='hue')
        fp = f"%s/%4d_%4d.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()


if __name__ == "__main__":
    foo = Foo()

    plt.figure(figsize=(12, 4))
    for x in range(0, 4800, 500):
        for y in range(0, 4800, 500):
            foo.plot_time_series(x, y)

    i = 0
