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

        # # Check if saved data exists
        # self.__is_data_saved = False
        # if os.path.exists(os.path.join(self.rootpath, 'ndvi.npy')) and \
        #         os.path.exists(os.path.join(self.rootpath, 'evi.npy')) and \
        #         os.path.exists(os.path.join(self.rootpath, 'dates.npy')):
        #     self.__data_saved = True
        #     self.load_data()
        # else:
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
                if "NDVI" in sds:
                    self.ndvi.append(arr)
                if "EVI" in sds:
                    self.evi.append(arr)

                sds = sds.replace(' ', '_')
                pth = f"{self.rootpath}/{sds}"
                if not os.path.exists(pth):
                    os.makedirs(pth)
                fp = f"{pth}/{dt}.png"
                if not os.path.exists(fp):
                    print(f"plotting map {fp}", end=" ")
                    self.__plot_map(arr, sds, fp)
                else:
                    print(f"skipping plot {fp}", end=" ")
                print()

            hdf.end()
        # self.save_data()

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
        df["ndvi"] = np.array(self.ndvi)[:, x, y]
        df["evi"] = np.array(self.evi)[:, x, y]
        sns.lineplot(data=df)
        fp = f"%s/%4d_%4d.png" % (pth, x, y)
        plt.savefig(fp)
        print(f"plot saved at {fp}")
        plt.clf()

        # plt.plot(self.dates, ts1, marker='o', linestyle='-', color='b')
        # plt.plot(self.dates, ts2, marker='o', linestyle='-', color='b')
        # plt.title(f"Time Series for Pixel ({x}, {y})")
        # plt.xlabel("Date")
        # plt.ylabel("Vegetation Indices")
        # plt.grid(True)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        # fp = f"%s/%4d_%4d.png" % (pth, x, y)
        # if not os.path.exists(fp):
        #     plt.savefig(fp)
        # plt.clf()

    def save_data(self):
        np.save(os.path.join(self.rootpath, 'ndvi.npy'), self.ndvi)
        # np.save(os.path.join(self.rootpath, 'evi.npy'), self.evi)
        np.save(os.path.join(self.rootpath, 'dates.npy'), self.dates)

    def load_data(self):
        self.ndvi = np.load(os.path.join(self.rootpath, 'ndvi.npy'), allow_pickle=True).tolist()
        self.evi = np.load(os.path.join(self.rootpath, 'evi.npy'), allow_pickle=True).tolist()
        self.dates = np.load(os.path.join(self.rootpath, 'dates.npy'), allow_pickle=True).tolist()


if __name__ == "__main__":
    foo = Foo()

    plt.figure(figsize=(12, 6))
    for x in range(0, 5000, 1000):
        for y in range(0, 5000, 1000):
            foo.plot_time_series(x, y)

    i = 0
