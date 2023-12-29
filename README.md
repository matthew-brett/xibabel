# Experiments with Xarray and Nibabel

Work for integrating Xarray into Nibabel.

For the moment, let's work outside Nibabel.

Perhaps this package will stay outside Nibabel, let's see how it goes.

## Install

```
pip install flit
git clone https://github.com/matthew-brett/xibabel
cd xibabel
flit install
```

## Testing data

Some of the example notebooks refer to data that are available as git
submodules, so you'll need to grab those.

```
git submodule update --init
```
