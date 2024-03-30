# `xarray-ome-ngff`

Integrating [Xarray](https://docs.xarray.dev/en/stable/) with [OME-NGFF](https://ngff.openmicroscopy.org/)

# Usage

## Read an OME-NGFF Zarr group as a dict of `xarray.DataArray`
```python
import zarr
from xarray_ome_ngff import read_multiscale_group, DaskArrayWrapper
group = zarr.open_group("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr")
arrays = read_multiscale_group(group, array_wrapper=DaskArrayWrapper(chunks=10))
print(arrays)
"""
{'0': <xarray.DataArray 'array-bb42996937dbff7600e0481e2b1572cc' (c: 2, z: 236,
                                                            y: 275, x: 271)>
dask.array<array, shape=(2, 236, 275, 271), dtype=uint16, chunksize=(2, 10, 10, 10), chunktype=numpy.ndarray>
Coordinates:
  * c        (c) float64 0.0 1.0
  * z        (z) float64 0.0 0.5002 1.0 1.501 2.001 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 0.0 0.3604 0.7208 1.081 ... 97.67 98.03 98.39 98.75
  * x        (x) float64 0.0 0.3604 0.7208 1.081 ... 96.23 96.59 96.95 97.31, '1': <xarray.DataArray 'array-2bfe6d4a6d289444ca93aa84fcb36342' (c: 2, z: 236,
                                                            y: 137, x: 135)>
dask.array<array, shape=(2, 236, 137, 135), dtype=uint16, chunksize=(2, 10, 10, 10), chunktype=numpy.ndarray>
Coordinates:
  * c        (c) float64 0.0 1.0
  * z        (z) float64 0.0 0.5002 1.0 1.501 2.001 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 0.0 0.7208 1.442 2.162 ... 95.87 96.59 97.31 98.03
  * x        (x) float64 0.0 0.7208 1.442 2.162 ... 94.42 95.15 95.87 96.59, '2': <xarray.DataArray 'array-80c5fc67c0c57909c0a050656a5ab630' (c: 2, z: 236,
                                                            y: 68, x: 67)>
dask.array<array, shape=(2, 236, 68, 67), dtype=uint16, chunksize=(2, 10, 10, 10), chunktype=numpy.ndarray>
Coordinates:
  * c        (c) float64 0.0 1.0
  * z        (z) float64 0.0 0.5002 1.0 1.501 2.001 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 0.0 1.442 2.883 4.325 5.766 ... 92.26 93.7 95.15 96.59
  * x        (x) float64 0.0 1.442 2.883 4.325 5.766 ... 90.82 92.26 93.7 95.15}
"""
```