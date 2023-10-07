import rasterio as rio  # the GEOS-based raster package
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime, timedelta


class Foo:
    def __init__(self):
        self.rootpath = './h12v04'
        self.hdf_files = [f for f in os.listdir(self.rootpath) if f.endswith('.hdf')]
        self.ndvi_data = []
        self.dates = []
        pass

    @staticmethod
    def extract_date_from_filename(filename):
        year = int(filename[9:13])
        doy = int(filename[13:16])  # day of year
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        return date.strftime('%Y-%m-%d')

    def foo(self):
        for hdf_file in self.hdf_files:
            dt = self.extract_date_from_filename(hdf_file)
            print(dt)
            self.dates.append(dt)

            file_path = os.path.join(self.rootpath, hdf_file)

            hdf = SD(file_path, SDC.READ)
            for idx, sds in enumerate(hdf.datasets().keys()):
                if "NDVI" in sds:
                    print(hdf_file, sds)
                    ds = hdf.select(sds)
                    self.ndvi_data.append(ds[:])

                    # self.plot_map(arr, sds, hdf_file)
            hdf.end()

    def plot_map(self, arr, sds, hdf_file):
        plt.figure(figsize=(10, 10))
        plt.imshow(arr, cmap='RdYlGn')
        plt.colorbar(label=sds)
        plt.title(sds)
        dst_file = sds.replace(" ", "_") + ".png"
        dst_folder = self.rootpath + "/" + hdf_file.rsplit(".", 1)[0] + "/"
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        plt.savefig(dst_folder + dst_file)
        plt.clf()

    def plot_time_series(self, x, y):
        ts = np.array(foo.ndvi_data)[:, x, y]
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, ts, marker='o', linestyle='-', color='b')
        plt.title(f"Time Series for Pixel ({x}, {y})")
        plt.xlabel("Date")
        plt.ylabel("Average NDVI")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.rootpath, "NDVI_time_series.png"))

    @staticmethod
    def _extract_date_from_filename(filename):
        """Extract date information from the filename."""
        match = re.search(r'A(\d{4})(\d{3})', filename)
        if match:
            year = match.group(1)
            day_of_year = match.group(2)
            return f"{year}-{day_of_year}"
        return "Unknown-Date"


if __name__ == "__main__":
    foo = Foo()
    foo.foo()
    foo.plot_time_series(0, 0)
