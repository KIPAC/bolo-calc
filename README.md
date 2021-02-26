# The bolo-calc package

## Set-up and testing
Setup from bash
```
git clone https://github.com/KIPAC/bolo-calc.git
cd bolo-calc
python setup.py install (or python setup.py develop)
```

Download atmosphere file
```
scripts/update_atm.py
```

Running unit tests from bash (from top-level bolo-calc directory)
```
pip install pytest
py.test
```

## Demo

Running script (from top-level bolo-calc directory)
```
scripts/bolo-calc.py -i config/myExample.py -o test.fits
```

Running jupyter notebook (from top-level bolo-calc directory)
```
jupyter-notebook nb/bolo-example.ipynb
```


## People
* [Eric Charles](https://github.com/KIPAC/bolo-calc/issues/new?body=@eacharles) (SLAC/KIPAC)

## License, etc.

This is open source software, available under the BSD license. If you are interested in this project, please do drop us a line via the hyperlinked contact names above, or by [writing us an issue](https://github.com/KIPAC/bolo-calc/issues/new).
