a = [0, 1, 12, 3]
print(max(*a))

def f():
    np = object()
    nib_img = object()
    slicers = []
    new_affine = 0
    assert np.all(nib_img.slicer[*slicers].affine == new_affine)
