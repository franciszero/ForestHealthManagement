import rasterio as rio  # the GEOS-based raster package
from rasterio.plot import show
from PIL import Image


with rio.open(r"../ABoVE/Bh14v15/Burn_Scar_dNBR/ABoVE.BID.1990.Bh14v15.003.2018117180506.tif") as dataset:
    print(dataset.profile)
    band1 = dataset.read(1)
    # show(band1)
    for band_num in range(1, dataset.count + 1):
        band_data = dataset.read(band_num)
        image = Image.fromarray(band_data)
        image.save(f'band{band_num}.png')
    print(dataset.bounds)
    print(dataset.crs)
