# Making a release

Update the version number in `src/xibabel/__init__.py`.

```bash
pip install build twine
```

For reassurance:

```bash
git clean -fxd
```

Build the Sdist:

```bash
python -m build --sdist
```

Upload to PyPI:

```python
twine upload dist/xibabel*.tar.gz
```

## Documentation

```{python}
make gh-pages
```
