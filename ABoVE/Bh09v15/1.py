import rasterio as rio  # the GEOS-based raster package
from rasterio.plot import show
from PIL import Image


with rio.open(r"ABoVE.AirSWOT_Radar.2017229170327.Ch089v106.001.2018139050250.tif") as dataset:
    print(dataset.profile)
    band1 = dataset.read(1)
    # show(band1)
    for band_num in range(1, dataset.count + 1):
        band_data = dataset.read(band_num)
        image = Image.fromarray(band_data)
        image.save(f'band{band_num}.png')
    print(dataset.bounds)
    print(dataset.crs)
