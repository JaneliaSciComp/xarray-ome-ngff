# xarray-ome-ngff

Integrating [Xarray](https://docs.xarray.dev/en/stable/) with [OME-NGFF](https://ngff.openmicroscopy.org/).

## Help

See [documentation](https://janeliscicomp.github.io/xarray-multiscale) for more details

## Usage

### Read OME-NGFF data

```python
import zarr
from xarray_ome_ngff import read_multiscale_group, DaskArrayWrapper
group = zarr.open_group("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr")

# this ensures that we create a Dask array, which gives us lazy loading
array_wrapper = DaskArrayWrapper(chunks=10)
arrays = read_multiscale_group(group, array_wrapper=array_wrapper)
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

### Create OME-NGFF data

```python
import numpy as np
from xarray import DataArray
from xarray_ome_ngff import create_multiscale_group
from zarr import MemoryStore

base_array = DataArray(
  np.zeros((10,10), dtype='uint8'),
  coords={
    'x': DataArray(np.arange(-5,5) * 3, dims=('x',), attrs={'units': 'meter'}),
    'y': DataArray(np.arange(-10, 0) * 3, dims=('y',), attrs={'units': 'meter'})
    })

# create a little multiscale pyramid
arrays = {
  's0': base_array,
  's1': base_array.coarsen({'x': 2, 'y': 2}, boundary='trim').mean().astype(base_array.dtype)
}

# This example uses in-memory storage, but you can use a 
# different store class from `zarr`
store = MemoryStore()

group = create_multiscale_group(store=store, path='my_group', arrays=arrays)
print(group.attrs.asdict())
"""
{
    'multiscales': (
        {
            'version': '0.4',
            'name': None,
            'type': None,
            'metadata': None,
            'datasets': (
                {
                    'path': 's0',
                    'coordinateTransformations': (
                        {'type': 'scale', 'scale': (3.0, 3.0)},
                        {'type': 'translation', 'translation': (-15.0, -30.0)},
                    ),
                },
                {
                    'path': 's1',
                    'coordinateTransformations': (
                        {'type': 'scale', 'scale': (6.0, 6.0)},
                        {'type': 'translation', 'translation': (-13.5, -28.5)},
                    ),
                },
            ),
            'axes': (
                {'name': 'x', 'type': 'space', 'unit': 'meter'},
                {'name': 'y', 'type': 'space', 'unit': 'meter'},
            ),
            'coordinateTransformations': None,
        },
    )
}
"""

# check that the arrays are there
print(tuple(group.arrays()))
"""
(('s0', <zarr.core.Array '/my_group/s0' (10, 10) uint8>), ('s1', <zarr.core.Array '/my_group/s1' (5, 5) uint8>))
"""

# write data to the arrays
for path, array in arrays.items():
  group[path][:] = array.data
```