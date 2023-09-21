# Links

## Surface and coordinate image formats

* [Coordinate Image API proposal](https://nipy.org/nibabel/devel/biaps/biap_0009.html)
* [Surface API meeting notes](https://hackmd.io/ZXcVpr1wQvmQIq9Sl1Vidg)
* [Surface formats meeting plan and notes](https://github.com/orgs/open-dicom/discussions/3)
* [Preliminary notes on surface formats](https://github.com/orgs/nipy/discussions/2)
* [Chris' and Matt's ITK / Nibabel discussion](https://demo.hedgedoc.org/VlVvlbOIS2qbDHIzXteKIw#)
* [BIDs spaces and mappings proposal](https://docs.google.com/document/d/11gCzXOPUbYyuQx8fErtMO9tnOKC3kTWiL9axWkkILNE)

## Spatial image formats

* [Planning and some notes for general image format meeting](https://github.com/orgs/open-dicom/discussions/1)
* See Chris' and Matt's ITK / Nibabel discussion above.
* [Planning for data formats mailing list
  discussion](https://mail.python.org/pipermail/neuroimaging/2021-November/002365.html)
* [BIDS proposed multidimensional array
  format](https://github.com/bids-standard/bids-specification/issues/197)
* [Neurodata Without Borders format](https://nwb-overview.readthedocs.io)
  — "Neurodata Without Borders (NWB) provides a common self-describing
  format ideal for archiving neurophysiology data and sharing it with
  colleagues." — "An NWB file is an HDF5 file that is structured in
  a particular way and has the .nwb file extension."
* [NeuroJson](http://neurojson.org).
* See below for the ASDF format, and HDF5

### HDF5

* [Moving away from HDF5](https://cyrille.rossant.net/moving-away-hdf5),
  followed by [should you use
  HDF5?](https://cyrille.rossant.net/should-you-use-hdf5/)
* [HDF5 for
  neuroimaging doc (Anderson Winkler)](https://docs.google.com/document/d/1s5DX4YPS680mc3Rb9msLetrjlPcDhaw835Um-jJb-Dw)
* [Storing NIFTI as HDF5 doc (Anderson
  Winkler)](https://docs.google.com/document/d/1hL27J2wNqHj27aX3VrCY8ajHyIBYtgYVmNmSoZZC8aA).


## Next-Generation File Formats (NGFF)

Bioimaging file format focused on microscopy.

* [NGFF](https://ngff.openmicroscopy.org) — "OME-NGFF is an imaging format
  specification being developed by the bioimaging community to address
  issues of scalability and interoperability."
* [Discussion threads on NGFF](https://forum.image.sc/tag/ome-ngff)
* [NGFF working group on
  metadata](https://quarep.org/working-groups/wg-7-metadata).

## Python imaging data formats

* zarr — see below
* xarray
* [fsspec](https://filesystem-spec.readthedocs.io/en/latest) - "Filesystem
  Spec (fsspec) is a project to provide a unified pythonic interface to
  local, remote and embedded file systems and bytes storage."
* [Intake](https://intake.readthedocs.io/en/latest/) — "Intake is
  a lightweight package for finding, investigating, loading and
  disseminating data."

## Zarr

See [Zarr.dev](https://zarr.dev) — "Zarr is a community project to develop
specifications and software for storage of large N-dimensional typed arrays,
also commonly known as tensors.".

But — this is a bit deceptive.  Zarr is a library that provides an API to
array-like things that can live on a wide range of storage backends — and
which you can slice like ordinary Numpy arrays: "Zarr provides classes and
functions for working with N-dimensional arrays that behave like NumPy
arrays but whose data is divided into chunks and each chunk is compressed."
[Zarr tutorial](https://zarr.readthedocs.io/en/stable/tutorial.html). It
defines a protocol to use for reading and writing such arrays — a *backend*
protocol.  It also defines a default backend and storage format.  This
specific storage format is what the docs mean when they say: "Zarr is a file
storage format for chunked, compressed, N-dimensional arrays based on an
open-source specification." — [Zarr docs](https://zarr.readthedocs.io).  But
you can also use the Zarr API to write to many other formats, or a format
you define yourself.

The Zarr default storage format, is currently in version 2, but soon to be
update to version 3. The [Zarr V3
spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)

The Zarr API conceives data as
a [hierarchy](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#hierarchy),
very much like the groups and arrays in HDF5 (and you can use HDF5 as a Zarr
backend).

> A Zarr hierarchy is a tree structure, where each node in the tree is
either a group or an array. Group nodes may have children but array nodes
may not. All nodes in a hierarchy have a name and a path. The root of a Zarr
hierarchy may be either a group or an array. In the latter case, the
hierarchy consists of just the single array.

There are many other [storage formats than Zarr can
use](https://zarr.readthedocs.io/en/stable/tutorial.html#storage-alternatives).

## ASDF

* [ASDF format
  paper](https://www.sciencedirect.com/science/article/pii/S2213133715000645)
  — an example of a format where binary data is binary, but metadata is
  text, and human-readable.
* [Docs for ASDF](https://asdf-standard.readthedocs.io/en/latest/)
* [Zarr backend for
  ASDF](https://github.com/braingram/asdf_zarr/tree/deferred_block)
* [ASDF plugin for Zarr](https://github.com/asdf-format/asdf-zarr)
* [Zarr / ASDF issue](https://github.com/asdf-format/asdf/issues/718)

## Other formats

* [ISMRM raw data format](https://ismrmrd.github.io) — "... a common raw
  data format [for MRI], which attempts to capture the data fields that are
  required to describe the magnetic resonance experiment with enough detail
  to reconstruct images.

## Metadata

* [Discussion of metadata preservation](https://github.com/orgs/open-dicom/discussions/8)
* [Importance or otherwise of virtual memory for image format](https://github.com/orgs/open-dicom/discussions/4)
* [Image compression](https://github.com/orgs/open-dicom/discussions/2)

## Dask

[Distributed computing for arrays](https://docs.dask.org/en/stable/index.html):

> Dask is a flexible library for parallel computing in Python.
>
> Dask is composed of two parts:
>
> Dynamic task scheduling optimized for computation. This is similar to
Airflow, Luigi, Celery, or Make, but optimized for interactive computational
workloads.
>
> “Big Data” collections like parallel arrays, dataframes, and lists that
extend common interfaces like NumPy, Pandas, or Python iterators to
larger-than-memory or distributed environments. These parallel collections run
on top of dynamic task schedulers.

## Xarray

[Xarray](https://docs.xarray.dev).

> Xarray introduces labels in the form of dimensions, coordinates and attributes on top of raw NumPy-like multidimensional arrays, which allows for a more intuitive, more concise, and less error-prone developer experience.

[Xarray and Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html)

[Xarray and
Zarr](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html)
