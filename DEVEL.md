# Development Notes

The development branch is ```devel``` where the last features are available but it could (should, surely!) have bugs.

To develop ELIANA do the usual stuff:

```bash
git clone https://github.com/paranal-sw/eliana
git checkout devel
cd eliana
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

## Unit tests

TODO, for now.

# Build and PyPI

# Build the distribution packages
python -m build

# Upload to PyPI
twine upload dist/*
