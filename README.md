# Xibabel

Xibabel is an experimental package for working with neuroimaging data formats.

It builds on the standard Nibabel package, but adds the extensions listed below.

The extensions allow code like this:

## Quickstart

Here is a basic read and slice operation with Xibabel. Compare to the more manual labor that you would have to do with Nibabel.

To run this code, install Xibabel with:

```bash
pip install --pre xibabel[optional]
```

```python
# Basic read and slice with Xibabel
import xibabel as xib

import matplotlib.pyplot as plt
plt.rc('image', cmap='gray')

# We can load NIfTI images directly from web URLs.
# This is a 4D (functional image)
ximg = xib.load('https://s3.amazonaws.com/openneuro.org/ds000105/sub-1/func/sub-1_task-objectviewing_run-01_bold.nii.gz')

# Slicing can now use axis labels:
mean_img = ximg.mean('time')

# Notice that we haven't yet fetched the data.   We do so only when we need
# it - for example, when plotting the image data:
plt.imshow(mean_img.sel(k=32))
```

See [Xibabel documentation](https://matthew-brett.github.io/xibabel) for more.

## Features

- Xibabel images are [Xarrays](https://docs.xarray.dev). They have labeled
  axes, with default labels for spatial axes of `i`, `j` and `k`; time is
  `time`. The labels allow slicing operations such as selecting slices over
  time.
- The labels allow concise and readable operations on named axes.
  See the [glm_in_xibabel
  notebook](https://matthew-brett.github.io/xibabel/glm_with_xibabel) for an example.
- Xarrays have attributes that can be attached to the image or the axes of the
  image. We can load these attributes from
  [BIDS](https://bids.neuroimaging.io/) format JSON files. This allows better
  transmission of metadata.
- The Xarrays have a Dask backend, so computations can be deferred and run at
  the point at which you need the data in memory.
- Xarrays and Dask allow new storage formats, including storage as
  [Zarr](https://zarr.readthedocs.io) and
  HDF5 / [netCDF](https://en.wikipedia.org/wiki/NetCDF).
- You can optimize the on-disk format for memory and CPU by adjusting
  _chunking_. We are working on performance metrics for different processing
  steps.
- Xibabel uses [fsspec](https://filesystem-spec.readthedocs.io) for reading
  NIfTI and other files, allowing you to use many filesystems as the source for
  your data, including HTTP, Amazon, Google Cloud and others. See the FSSpec
  documentation for details, and the code above for an example using HTTP as
  a backing store.

## Status

Xibabel is in development mode at the moment. We are still experimenting with the API. We'd love to hear from you if you are interested to help. Please do not rely on any particular features in this alpha version, including compatibility of file formats; prefer

## Install

From Pip — the current pre-release:

```bash
pip install --pre xarray
```

From Pip — the development code:

```bash
pip install git+https://github.com/matthew-brett/xibabel@main
```

## License

We release Xibabel under a BSD simplified 2-clause license (see `LICENSE`).

## Acknowledgments

This work was entirely supported by Chan Zuckerberg Initiative Essential Open
Source Software for Science grant "Strengthening community and code foundations
for brain imaging", with thanks.
