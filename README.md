# xarray-ome-ngff
Integration between xarray and the ome-ngff data model.

At present (February, 2023) this is a partial implementation of the [OME-NGFF spec](https://ngff.openmicroscopy.org/latest/#implementations). Specifcally, *only* the [`multiscales`](https://ngff.openmicroscopy.org/latest/#multiscale-md) and specs required by `multiscales` are implemented. Complete support for the spec would be welcome.

This library provides an extension of the OME-NGFF spec, called `OME-NGFF-J`, which adds `axes` and `coordinateTransformations` metadata to datasets inside a group bearing `multiscales` metadata. Hopefully future versions of the OME-NGFF spec adopt this convention or a compatible one, and this extension can be absorbed or replaced.
