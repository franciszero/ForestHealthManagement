import rasterio as rio  # the GEOS-based raster package
import matplotlib.pyplot as plt


with rio.open(r"../ABoVE/Bh14v15/NDVI_Trends_from_Landsat_1984-2012/ABoVE_NDVI_trend_sig_Ah002v002_Bh014v015.tif") as dataset:
    print(dataset.profile)
    band1 = dataset.read(1)
    # show(band1)
    for band_num in range(1, dataset.count + 1):
        band_data = dataset.read(band_num)

        plt.figure(figsize=(10, 10))
        plt.imshow(band1, cmap="RdYlBu", vmin=band_data.min(), vmax=band_data.max())  # RdYlBu color mapping
        plt.colorbar(label="")
        plt.title("dNBR Visualization")
        plt.show()
    print(dataset.bounds)
    print(dataset.crs)
