# xarray-ome-ngff
Integration between xarray and the ome-ngff data model.

At present (February, 2023) this is a partial implementation of the [OME-NGFF spec](https://ngff.openmicroscopy.org/latest/#implementations). Specifcally, *only* the [`multiscales`](https://ngff.openmicroscopy.org/latest/#multiscale-md) and specs required by `multiscales` are implemented. Complete support for the spec would be welcome.

## How it works
This library depends on [`pydantic-ome-ngff`](https://github.com/JaneliaSciComp/pydantic-ome-ngff) which implements OME-NGFF metadata as [pydantic](https://docs.pydantic.dev/) models. 
[`Axes`](https://ngff.openmicroscopy.org/latest/#axes-md) metadata is inferred from a DataArray by iterating over the dimensions of the array and checking for `units` and `type` properties in the attributes of the `coords` assigned to each dimension. Dimensions without coordinates will raise an exception. Scale and translation `CoordinateTransforms` are inferred by inspecting the values of the coordinates for each dimension. Be advised that no attempt is made to verify that arrays are sampled on a regular grid.

## Usage

Generate `multiscales` metadata from a multiscale collection of DataArrays.

```python
from xarray import DataArray
import numpy as np
from xarray_ome_ngff import create_multiscale_metadata
import json
coords = {'z' : DataArray(np.arange(100), attrs={'units': 'nm', 'type': 'space'}, dims=('z',)),
         'y' : DataArray(np.arange(300) * 2.2, attrs={'units': 'nm', 'type': 'space'}, dims=('y')),
         'x' : DataArray((np.arange(300) * .5) + 1, attrs={'units': 'nm', 'type': 'space'}, dims=('x',))}

s0 = DataArray(data=0, coords=coords, dims=('z','y','x'), name='s0')
s1 = s0.coarsen({dim: 2 for dim in s0.dims}).mean()
s1.name = 's1'
# create a small multiscale pyramid
multiscale = [s0, s1]
metadata = create_multiscale_metadata(name='test', type='yes', arrays=multiscale)
print(metadata.json(indent=2))
```
```json
{
  "version": "0.5-dev",
  "name": "test",
  "type": "yes",
  "metadata": null,
  "datasets": [
    {
      "path": "s0",
      "coordinateTransformations": [
        {
          "type": "scale",
          "scale": [
            1.0,
            2.2,
            0.5
          ]
        },
        {
          "type": "translation",
          "translation": [
            0.0,
            0.0,
            1.0
          ]
        }
      ]
    },
    {
      "path": "s1",
      "coordinateTransformations": [
        {
          "type": "scale",
          "scale": [
            2.0,
            4.4,
            1.0
          ]
        },
        {
          "type": "translation",
          "translation": [
            0.5,
            1.1,
            1.25
          ]
        }
      ]
    }
  ],
  "axes": [
    {
      "name": "z",
      "type": "space",
      "units": null
    },
    {
      "name": "y",
      "type": "space",
      "units": null
    },
    {
      "name": "x",
      "type": "space",
      "units": null
    }
  ],
  "coordinateTransformations": [
    {
      "type": "scale",
      "scale": [
        1.0,
        1.0,
        1.0
      ]
    }
  ]
}
```

It is not possible to create a DataArray from OME-NGFF metadata, but together the OME-NGFF [`Axes`](https://ngff.openmicroscopy.org/latest/#axes-md) and [`CoordinateTransformations`](https://ngff.openmicroscopy.org/latest/#trafo-md) metadata are sufficient to create _coordinates_ for a DataArray, provided you know the shape of the data. The function `create_coords` performs this operation:

```python
from xarray_ome_ngff import create_coords
from pydantic_ome_ngff.v05.coordinateTransformations import VectorScaleTransform, VectorTranslationTransform
from pydantic_ome_ngff.v05.axes import Axis


shape = (3, 3)
axes = [Axis(name='a', units="meter", type="space"), Axis(name='b', units="meter", type="space")]

transforms = [VectorScaleTransform(scale=[1, .5]), VectorTranslationTransform(translation=[1, 2])]

coords = create_coords(axes, transforms, shape)
print(coords)
'''
{'a': <xarray.DataArray (a: 3)>
array([1., 2., 3.])
Dimensions without coordinates: a
Attributes:
    units:    meter, 'b': <xarray.DataArray (b: 3)>
array([2. , 2.5, 3. ])
Dimensions without coordinates: b
Attributes:
    units:    meter}
'''
```