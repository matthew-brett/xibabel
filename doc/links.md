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

"Zarr is a file storage format for chunked, compressed, N-dimensional arrays
based on an open-source specification." — [Zarr
docs](https://zarr.readthedocs.io).

* [Zarr V3
  spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)

## Other formats

* [ISMRM raw data format](https://ismrmrd.github.io) — "... a common raw
  data format [for MRI], which attempts to capture the data fields that are
  required to describe the magnetic resonance experiment with enough detail
  to reconstruct images.

## Metadata

* [Discussion of metadata preservation](https://github.com/orgs/open-dicom/discussions/8)
* [Importance or otherwise of virtual memory for image format](https://github.com/orgs/open-dicom/discussions/4)
* [Image compression](https://github.com/orgs/open-dicom/discussions/2)

## ASDF

* [ASDF format
  paper](https://www.sciencedirect.com/science/article/pii/S2213133715000645)
  — an example of a format where binary data is binary, but metadata is
  text, and human-readable.
* [Docs for ASDF](https://asdf-standard.readthedocs.io/en/latest/)
