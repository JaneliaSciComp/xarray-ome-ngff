# `xarray-ome-ngff`

Integrating [Xarray](https://docs.xarray.dev/en/stable/) with [OME-NGFF](https://ngff.openmicroscopy.org/).

# Motivation

Measurements without context are useless. In imaging applications we prevent our images from 
being useless by tracking where, and how, our images were acquired. This context makes images interpretable, which makes those images potentially useful. 

The [OME-NGFF](https://ngff.openmicroscopy.org/) project defines a very nice [Zarr](https://zarr.readthedocs.io/en/stable/)-based file 
format for storing images and spatial metadata, i.e. useful context about where the images
were acquired. `Xarray` defines a solid data model and python API for rasterized data with annotated coordinates. This package bridges OME-NGFF and Xarray by providing tools for reading and writing OME-NGFF images using Xarray data structures.

# Examples

## Read an OME-NGFF Zarr group as a dict of `xarray.DataArray`

This example accesses a public OME-NGFF multiscale image, which is stored in a Zarr group that 
contains several Zarr arrays, one per level of detail.
The [`read_multiscale_group`](api/index.md#xarray_ome_ngff.read_multiscale_group) function takes a 
reference to that OME-NGFF group and returns a dictionary of [`xarray.DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html), one per Zarr array contained in the multiscale group. 
OME-NGFF coordinate metadata (axis names, units, scaling and translation parameters) are read from 
the attributes of the Zarr group to generate coordinates for each `DataArray`.

```python
import zarr
from xarray_ome_ngff import read_multiscale_group, DaskArrayWrapper
group = zarr.open_group("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr")

# this ensures that we create a Dask array, which gives us lazy loading
array_wrapper = DaskArrayWrapper(chunks=10)
arrays = read_multiscale_group(group, array_wrapper=array_wrapper)
print(arrays)
"""
{'0': <xarray.DataArray 'array-dc1746969a43c4e18b4a861ae8a5b5e5' (c: 2, z: 236,
                                                            y: 275, x: 271)> Size: 70MB
dask.array<array, shape=(2, 236, 275, 271), dtype=uint16, chunksize=(2, 10, 10, 10), chunktype=numpy.ndarray>
Coordinates:
  * c        (c) float64 16B 0.0 1.0
  * z        (z) float64 2kB 0.0 0.5002 1.0 1.501 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 2kB 0.0 0.3604 0.7208 1.081 ... 97.67 98.03 98.39 98.75
  * x        (x) float64 2kB 0.0 0.3604 0.7208 1.081 ... 96.23 96.59 96.95 97.31, '1': <xarray.DataArray 'array-9663bb2b4241268352500eb4e3bef581' (c: 2, z: 236,
                                                            y: 137, x: 135)> Size: 17MB
dask.array<array, shape=(2, 236, 137, 135), dtype=uint16, chunksize=(2, 10, 10, 10), chunktype=numpy.ndarray>
Coordinates:
  * c        (c) float64 16B 0.0 1.0
  * z        (z) float64 2kB 0.0 0.5002 1.0 1.501 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 1kB 0.0 0.7208 1.442 2.162 ... 95.87 96.59 97.31 98.03
  * x        (x) float64 1kB 0.0 0.7208 1.442 2.162 ... 94.42 95.15 95.87 96.59, '2': <xarray.DataArray 'array-dd59cfa6aeac755c5fd9b53fd121f3b8' (c: 2, z: 236,
                                                            y: 68, x: 67)> Size: 4MB
dask.array<array, shape=(2, 236, 68, 67), dtype=uint16, chunksize=(2, 10, 10, 10), chunktype=numpy.ndarray>
Coordinates:
  * c        (c) float64 16B 0.0 1.0
  * z        (z) float64 2kB 0.0 0.5002 1.0 1.501 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 544B 0.0 1.442 2.883 4.325 ... 92.26 93.7 95.15 96.59
  * x        (x) float64 536B 0.0 1.442 2.883 4.325 ... 90.82 92.26 93.7 95.15}
"""
```

Note this code from the above example:

```{.python .test="skip"}
array_wrapper = DaskArrayWrapper(chunks=10)
arrays = read_multiscale_group(group, array_wrapper=array_wrapper)
```
The `DaskArrayWrapper` ensures that we wrap the Zarr arrays with the distributing computing library [`dask`](https://www.dask.org/). More information about this can be found in the documentation for [array wrapping](#array-wrapping).

## Read a Zarr array as a `DataArray`

The following example demonstrates how to access a single Zarr array as an `DataArray`. This 
process is more indirect than the previous example, thanks to some unfortunate 
design decisions in OME-NGFF. Specifically, OME-NGFF doesn't use Zarr array metadata for storing 
metadata about arrays.
Instead, that information is stored in some Zarr group that, directly or indirectly, contains the 
arrays. The OME-NGFF metadata that describes a given Zarr array cannot be discovered in advance; an 
application must search the Zarr hierarchy for metadata that describes the array, and hopefully 
there is only one such piece of metadata -- the OME-NGFF allows multiple, inconsistent descriptions 
of the same Zarr array. This should be fixed, but it's too late for the current version of the spec 
(0.4). 

In any case, the following example works, but only
when the array is described by a single instance of OME-NGFF metadata (which is typical usage).

```python
import zarr
from xarray_ome_ngff import read_multiscale_array, DaskArrayWrapper
array_z = zarr.open_array("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr/2")
array_x = read_multiscale_array(array_z)
print(array_x)
"""
<xarray.DataArray (c: 2, z: 236, y: 68, x: 67)> Size: 4MB
array([[[[28, 10,  9, ...,  9,  9,  9],
         [10,  9,  9, ...,  9, 23, 10],
         [ 9,  9,  9, ...,  8, 93,  9],
         ...,
         [ 9,  8,  9, ...,  9,  9, 10],
         [10,  8, 10, ...,  9, 10, 10],
         [10,  9,  8, ...,  9,  9,  9]],

        [[10, 10, 10, ..., 10, 10, 10],
         [ 9,  9,  9, ...,  9, 10,  9],
         [ 8,  8,  9, ...,  8,  9,  9],
         ...,
         [ 9,  9, 10, ...,  9,  9,  9],
         [10, 10, 10, ..., 10, 11, 10],
         [10, 10, 10, ..., 10, 10, 10]],

        [[ 9,  8,  9, ..., 10, 10,  8],
         [ 9, 10,  9, ..., 10, 10,  9],
         [ 9,  8,  9, ...,  8, 40,  9],
         ...,
...
         ...,
         [28, 28, 28, ..., 28, 41, 28],
         [29, 28, 29, ..., 28, 28, 35],
         [28, 28, 28, ..., 28, 28, 33]],

        [[39, 27, 27, ..., 48, 28, 29],
         [28, 27, 29, ..., 43, 33, 32],
         [28, 28, 28, ..., 29, 29, 28],
         ...,
         [28, 28, 28, ..., 34, 42, 28],
         [29, 28, 29, ..., 29, 28, 41],
         [29, 29, 30, ..., 28, 28, 30]],

        [[28, 29, 28, ..., 30, 28, 28],
         [29, 28, 27, ..., 55, 33, 27],
         [27, 27, 27, ..., 27, 31, 27],
         ...,
         [28, 28, 28, ..., 28, 28, 28],
         [28, 27, 28, ..., 39, 28, 28],
         [28, 28, 28, ..., 27, 28, 28]]]], dtype=uint16)
Coordinates:
  * c        (c) float64 16B 0.0 1.0
  * z        (z) float64 2kB 0.0 0.5002 1.0 1.501 ... 116.0 116.5 117.0 117.5
  * y        (y) float64 544B 0.0 1.442 2.883 4.325 ... 92.26 93.7 95.15 96.59
  * x        (x) float64 536B 0.0 1.442 2.883 4.325 ... 90.82 92.26 93.7 95.15
"""
```

## Create a OME-NGFF Zarr group from a collection of `DataArray`

You can create OME-NGFF metadata from a dictionary with string keys 
and `DataArray` values.

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
# this will not work if your arrays are large!
for path, array in arrays.items():
  group[path][:] = array.data

```

# Details

## Inferring transform metadata from explicit coordinates

OME-NGFF models an image as if each sample corresponds to an element from a finite, regular, N-dimensional grid. In this model, which is common in bioimaging, each axis of the that grid is parametrized by 3 numbers: 

- The number of samples 
    - stored in the `shape` attribute of a Zarr array
- The separation between adjacent samples
    - stored in one or two instances of [`coordinateTransformations` metadata](https://ngff.openmicroscopy.org/0.4/index.html#trafo-md). 
    - If this value is stored in two places, the two intermediate values must be multiplied to get the final value.
- The location of the first sample 
    - stored in one or two instances of [`coordinateTransformations` metadata](https://ngff.openmicroscopy.org/0.4/index.html#trafo-md). 
    - If this value is stored in two places, the intermediate two values must be summed to get the final value.

By contrast, in the coordinate framework used by Xarray, each sample of an image can be associated with multiple elements from multiple irregular grids. To support this expressiveness, Xarray represents these grids explicitly, as arrays of values. So where OME-NGFF defines coordinates with 3 numbers per dimension, Xarray would conventionally use `N` numbers per dimension, where `N` is the number of samples along that dimension (but it is possible to use more coordinates than this).

Converting between the OME-NGFF and Xarray coordinate representations is straightforward, but it requires some care. Generating a translation transformation from Xarray-style coordinates is easy -- we just take the first element of each coordinate variable. But generating a scaling transformation can be delicate, because we have to compute the difference between two adjacent coordinate values, per axis, and this can introduce floating point arithmetic errors, which can be corrected by rounding. Thus, routines that generate OME-NGFF coordinate metadata from Xarray coordinates in `xarray-ome-ngff` (like the imaginatively named [`transforms_from_coords`](./api/v04/multiscale.md#xarray_ome_ngff.v04.multiscale.transforms_from_coords)) take a `transform_precision` keyword argument that controls how much rounding to apply when generating OME-NGFF transforms from Xarray coordinates. The default value of `transform_precision` is `None`, which results in no rounding at all.

## Array wrapping

This library uses [`zarr-python`](https://zarr.readthedocs.io/en/stable/) for access to Zarr 
arrays and groups. If you provide a Zarr array as the `data` parameter to `xarray.DataArray`,
`xarray` will attempt to load the entire array into memory. This is not ideal, let alone 
possible, for many Zarr arrays, and so this library needs to provide a layer of indirection on 
top of raw `zarr.Array` instances, at least until `zarr-python` adds a lazy array API. We solve this
problem by defining lightweight classes that have a single method, `wrap`, that takes a Zarr array
and returns something array-like. Functions in this library that use Zarr arrays to create `xarray.DataArray` objects take an `array_wrap` parameter that must be an instance of such a class (or a dict representation of such a class).

This library includes two such "array wrappers" -- one that passes
the input Zarr array through unaltered ([`ZarrArrayWrapper`](./api/array_wrap.md#xarray_ome_ngff.array_wrap.ZarrArrayWrapper)),
and a more interesting array wrapper that wraps a Zarr array in a [`dask.array.Array`](https://docs.dask.org/en/stable/array.html) (['DaskArrayWrapper`](./api/array_wrap.md#xarray_ome_ngff.array_wrap.DaskArrayWrapper)). The array wrappers are designed to be easy to implement; if the included wrappers are insufficient for an application, you should implement one that works for you.

## Extent of OME-NGFF support

Only the [image pyramid specification](https://ngff.openmicroscopy.org/latest/#multiscale-md) is supported by this package. Do you want more extensive OME-NGFF support in this library? 
Open an issue on the [issue tracker](https://github.com/JaneliaSciComp/xarray-ome-ngff)