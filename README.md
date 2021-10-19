# HYDROPT: a Python Framework for Fast Inverse Modelling of Multi- and Hyperspectral Ocean Color Data

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/tadz-io/hydropt)](https://github.com/tadz-io/hydropt/releases/latest)
[![license](https://img.shields.io/github/license/tadz-io/hydropt?label=license)](https://github.com/tadz-io/hydropt/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5086370.svg)](https://doi.org/10.5281/zenodo.5086370)
[![Python package](https://github.com/tadz-io/hydropt/actions/workflows/python-package.yml/badge.svg)](https://github.com/tadz-io/hydropt/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/tadz-io/hydropt/branch/master/graph/badge.svg?token=95Y1Z31F5C)](https://codecov.io/gh/tadz-io/hydropt)
## Description
<!-- start_ppi_description -->
HYDROPT is an open-source framework for forward and inverse modelling of multi- and hyperspectral observations from oceans, coastal and inland waters. The remote sensing reflectance, R<sub>rs</sub>, is calculated by specifying the inherent optical properties (IOP) of the water column, the sensor viewing geometry and solar zenith angle. Our framework is based on radiative transfer principles and is sensor agnostic allowing R<sub>rs</sub> to be calculated for any wavelength in the 400 - 710 *nm* range. 

Inversion of R<sub>rs</sub> spectra is achieved by minimizing the difference between the HYDROPT forward calculations and the reflectance measured by the sensor. Different optimization routines can be selected to minimize the cost function. An extensive description of the theoretical basis of the framework as well as applications are provided in the following scientific papers:

> Holtrop, T., & Van Der Woerd, H. J. (**2021**). HYDROPT: An Open-Source Framework for Fast Inverse Modelling of Multi- and Hyperspectral Observations from Oceans, Coastal and Inland Waters. *Remote Sensing*, 13(15), 3006. [doi:10.3390/rs13153006](https://www.mdpi.com/2072-4292/13/15/3006)

>Van Der Woerd, H.J. & Pasterkamp, R. (**2008**). HYDROPT: A fast and flexible method to retrieve chlorophyll-a from multispectral satellite observations of optically complex coastal waters. *Remote Sensing of Environment*, 112, 1795–1807. [doi:10.1016/j.rse.2007.09.001](https://www.sciencedirect.com/science/article/abs/pii/S003442570700421X?via%3Dihub)

<!-- stop_ppi_description -->

Please cite our latest publication if you decide to use HYDROPT in your research:

```
@article{
    Holtrop_2021,
    title={HYDROPT: An Open-Source Framework for Fast Inverse Modelling of Multi- and Hyperspectral Observations from Oceans, Coastal and Inland Waters},
    author={Holtrop, Tadzio and Van Der Woerd, Hendrik Jan},
    journal={Remote Sensing}, 
    volume={13},
    number={15}, 
    month={Jul}, 
    pages={3006},
    year={2021}, 
    DOI={10.3390/rs13153006}, 
    publisher={MDPI AG}
}
```

## Features

- Specification of IOP models for forward and inverse modelling
- Sensor agnostic calculations of R<sub>rs</sub> in 400 - 710 *nm* range
- Calculation of R<sub>rs</sub> at nadir; off-nadir angles will be implemented in the future
- Specification of solar zenith angle will be implemented in the future (30 degrees sza by default)
- Levenberg-Marquardt optimization is used for the inversion; future versions will be able to select the full range of optimization routines offered in ```SciPy``` and ```LMFIT``` libraries.
## Installation

Install HYDROPT using pip:

```bash
pip install hydropt-oc
```

## Getting started
An example of how to create a case-I bio-optical model and perform forward and inverse calculations. First import the HYDROPT framework:

```python
import hydropt.hydropt as hd
```

Let's run the forward and inverse calculations at every 5 nm between 400 and 710 nm. First specify the wavebands:

```python
import numpy as np

wavebands = np.arange(400, 711, 5)
```

We can import the inherent optical properties (IOP) of water from the ```bio_optics``` module and create an optical model for this component as follows:

```python
from hydropt.bio_optics import H2O_IOP_DEFAULT

def clear_nat_water(*args):
    return H2O_IOP_DEFAULT.T.values
```

Every optical component should be constructed as a Python function that returns the IOPs as a *2xn* numpy array where *n* is the number of wavebands. The first row (```arr[0]```) should list the absorption values, the second row (```arr[1]```) lists the backscatter values. For phytoplankton we define the optical model in a similair way, importing the absorption values from the ```bio_optics``` module and specifying a constant spectral backscatter:

```python
from hydropt.bio_optics import a_phyto_base_HSI

def phytoplankton(*args):
    chl = args[0]
    # basis vector - according to Ciotti&Cullen (2002)
    a = a_phyto_base_HSI.absorption.values
    # constant spectral backscatter with backscatter ratio of 1.4%
    bb = np.repeat(.014*0.18, len(a))

    return chl*np.array([a, bb])
```

For colored dissolved organic matter (CDOM) we do the following:

```python
def cdom(*args):
    # absorption at 440 nm
    a_440 = args[0]
    # spectral absorption
    a = np.array(np.exp(-0.017*(wavebands-440)))
    # no backscatter
    bb = np.zeros(len(a))

    return a_440*np.array([a, bb])
```

The IOPs of all optical components should be specified at the same wavebands. Now that all optical components are created lets add them to an instance of the ```BioOpticalModel``` class:

```python
bio_opt = hd.BioOpticalModel()
# set optical models
bio_opt.set_iops(
    wavebands=wavebands,
    water=clear_nat_water,
    phyto=phytoplankton,
    cdom=cdom)
```

It is important that the keyword for the water optical model argument is called ```water```. We can check if everything works correctly by plotting the mass specific IOPs for these components:

```python
bio_opt.plot(water=None, phyto=1, cdom=1)
```

Now we can initialize the HYDROPT forward model with the bio-optical model, ```bio_opt```, that we have just created and calculate R<sub>rs</sub> when the phytoplankton concentration is 0.15 mg/m<sup>3</sup> and CDOM absorption is 0.02 m<sup>-1</sup>:

```python
# the HYDROPT polynomial forward model
fwd_model = hd.PolynomialForward(bio_opt)
# calculate Rrs
rrs = fwd_model.forward(phyto=.15, cdom=.02)
```

Lets invert the R<sub>rs</sub> spectrum we just calculated with the specified forward model ```fwd_model```. At this point HYDROPT only supports the Levenberg-Marquardt routine from the ```LMFIT``` library (```lmfit.minimize```). Please refer to the [LMFIT documentation](https://lmfit.github.io/lmfit-py/) for more information. 

Specify an initial guess for the phytoplankton concentration and CDOM absorption used for the Levenberg-Marquardt routine. 

```python
import lmfit
# set initial guess parameters for LM
x0 = lmfit.Parameters()
# some initial guess
x0.add('phyto', value=.5)
x0.add('cdom', value=.01)
```
Now invert R<sub>rs</sub> to retrieve the concentration and absorption of phytoplankton and CDOM respectively:

```python
# initialize an inversion model
inv_model = hd.InversionModel(
    fwd_model=fwd_model,
    minimizer=lmfit.minimize)
# estimate parameters
xhat = inv_model.invert(y=rrs, x=x0)
```
That's it! 

## Documentation

Documentation will be available soon. For questions please reach out!
## License
[AGPL-3.0](./LICENSE)