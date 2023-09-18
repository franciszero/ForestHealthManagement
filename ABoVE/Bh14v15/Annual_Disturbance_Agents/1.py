import rasterio as rio  # the GEOS-based raster package
from rasterio.plot import show
from PIL import Image
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import from_origin
from datetime import datetime, timedelta


class Foo:
    def __init__(self):
        # 打开NetCDF文件
        """
        data source: https://search.earthdata.nasa.gov/search/granules?p=C2226005584-ORNL_CLOUD!C2162140027-ORNL_CLOUD&pg[1][v]=t&pg[1][gsk]=-start_date&pg[1][m]=download&g=G2234622683-ORNL_CLOUD&q=Canada%20greenness&as[science_keywords][0]=Biosphere%3AVegetation%3AVegetation%20Cover%3A%3A%3ANdvi&tl=1694313784!3!!&fst0=land%20surface&fst1=Biosphere&fsm1=Vegetation&fs11=Vegetation%20Cover&fsd1=Ndvi&lat=53.525346701956174&long=-122.03613281250001&zoom=5
        data file: ABoVE_ForestDisturbance_Agents.Bh14v15_dTC.nc
        time range: 1985 ~ 2012
        """
        self.dataset_name = "Bh14v15_dTC.nc"
        self.nc_file = nc.Dataset(self.dataset_name, 'r')
        # 读取变量数据
        self.year_id = self.nc_file.variables['year_id'][:]
        self.albers_conical_equal_area = self.nc_file.variables['albers_conical_equal_area']
        self.d_brightness = self.nc_file.variables['d_brightness'][:]
        self.d_greenness = self.nc_file.variables['d_greenness'][:]
        self.d_wetness = self.nc_file.variables['d_wetness'][:]
        self.pre_brightness = self.nc_file.variables['pre_brightness'][:]
        self.pre_greenness = self.nc_file.variables['pre_greenness'][:]
        self.pre_wetness = self.nc_file.variables['pre_wetness'][:]
        self.lat = self.nc_file.variables['lat'][:]
        self.lon = self.nc_file.variables['lon'][:]
        self.time = self.nc_file.variables['time'][:]
        self.dates = [datetime(1985, 1, 1) + timedelta(days=int(val)) for val in self.time]

        self.x = self.nc_file.variables['x'][:]
        self.y = self.nc_file.variables['y'][:]
        pass

    def plot_save(self, filename, slice_data):
        # # 计算总绿度
        # slice_data = np.where(slice_data == 32767, np.nan, slice_data).min()

        # # 设置正确的变换和CRS
        # transform = from_origin(self.x[0], self.y[-1], self.x[1] - self.x[0], self.y[1] - self.y[0])
        # crs = rio.crs.CRS.from_string(self.albers_conical_equal_area.spatial_ref, 'WKT')
        # # 使用 rasterio 显示数据
        # with rio.open(
        #     filename + ".tif",
        #     'w',
        #     driver='GTiff',
        #     height=masked_data.shape[0],
        #     width=masked_data.shape[1],
        #     count=1,
        #     dtype=masked_data.dtype,
        #     crs=crs,
        #     transform=transform,
        # ) as dst:
        #     dst.write(masked_data, 1)
        # show(masked_data, transform=transform, cmap='jet', title=filename)

        # 使用 matplotlib 绘制并保存图像
        plt.imshow(slice_data.data, cmap='jet', origin='lower')
        plt.colorbar()
        plt.title(filename)
        plt.savefig(filename + ".png")
        plt.close()  # 这将确保图像被保存后关闭，并不会显示出来

    def plot_show(self):
        plt.figure(figsize=(15, 15))
        ra = self.year_id.shape[0]
        # ra = 3
        # filename

        # # 绿度
        for i in range(ra):
            slice_data = self.d_greenness[i] + self.pre_greenness[i]
            var = 'd_greenness'
            print(f"{var}_{i}")
            self.plot_save(self.get_file_name(var, i), slice_data)

        # # 亮度
        for i in range(ra):
            slice_data = self.d_brightness[i] + self.pre_brightness[i]
            var = 'd_brightness'
            print(f"{var}_{i}")
            self.plot_save(self.get_file_name(var, i), slice_data)

        # # 湿度
        for i in range(ra):
            slice_data = self.d_wetness[i] + self.pre_wetness[i]
            var = 'd_wetness'
            print(f"{var}_{i}")
            self.plot_save(self.get_file_name(var, i), slice_data)
        self.nc_file.close()

    def get_file_name(self, var, i):
        return f"{self.dataset_name.split('.')[0]}_{self.dates[i]}_{var}_{i + 1}"


f = Foo()
f.plot_show()

nn = 0
