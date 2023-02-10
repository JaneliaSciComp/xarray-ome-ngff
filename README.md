# xarray-ome-ngff
Integration between xarray and the ome-ngff data model.

At present (February, 2023) this is a partial implementation of the [OME-NGFF spec](https://ngff.openmicroscopy.org/latest/#implementations). Specifcally, *only* the [`multiscales`](https://ngff.openmicroscopy.org/latest/#multiscale-md) and specs required by `multiscales` are implemented. Complete support for the spec would be welcome.

(in progress) This library provides an extension of the OME-NGFF spec, called `OME-NGFF-J`, which adds `axes` and `coordinateTransformations` metadata to datasets inside a group bearing `multiscales` metadata. Hopefully future versions of the OME-NGFF spec adopt this convention or a compatible one, and this extension can be absorbed or replaced.

## How it works
OME-NGFF metadata is represented as [pydantic](https://docs.pydantic.dev/) models. 
[`Axes`](https://ngff.openmicroscopy.org/latest/#axes-md) metadata is inferred from a DataArray by iterating over the dimensions of the array and checking for `units` and `type` properties in the attributes of the `coords` assigned to each dimension. Dimensions without coordinates will raise an exception. Scale and translation `CoordinateTransforms` are inferred by inspecting the values of the coordinates for each dimension. Be advised that no attempt is made to verify that arrays are sampled on a regular grid.

## Usage

Generate `multiscales` metadata from a multiscale collection of DataArrays.

```python
from xarray import DataArray
import numpy as np
from xarray_ome_ngff.latest import MultiscaleGroupMetadata
import json
coords = {'z' : DataArray(np.arange(100), attrs={'units': 'nm', 'type': 'space'}, dims=('z',)),
         'y' : DataArray(np.arange(300) * 2.2, attrs={'units': 'nm', 'type': 'space'}, dims=('y')),
         'x' : DataArray((np.arange(300) * .5) + 1, attrs={'units': 'nm', 'type': 'space'}, dims=('x',))}

s0 = DataArray(data=0, coords=coords, dims=('z','y','x'), name='s0')
s1 = s0.coarsen({dim: 2 for dim in s0.dims}).mean()
s1.name = 's1'
# create a small multiscale pyramid
multiscale = [s0, s1]
metadata = MultiscaleMetadata.fromDataArrays(name='test', type='yes', arrays=multiscale)
print(metadata.json(indent=2))
```
```json
{
  "version": "0.5-dev",
  "name": "test",
  "type": "yes",
  "metadata": null,
  "axes": [
    {
      "name": "z",
      "type": "space",
      "unit": "nm"
    },
    {
      "name": "y",
      "type": "space",
      "unit": "nm"
    },
    {
      "name": "x",
      "type": "space",
      "unit": "nm"
    }
  ],
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
  "coordinateTransformations": null
}
```
