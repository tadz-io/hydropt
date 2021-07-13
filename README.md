# HYDROPT: a Python Framework for Fast Inverse Modelling of Multi- and Hyperspectral Ocean Color Data

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/tadz-io/hydropt)](https://github.com/tadz-io/hydropt/releases/latest)
[![license](https://img.shields.io/github/license/tadz-io/hydropt?label=license)](https://github.com/tadz-io/hydropt/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5086370.svg)](https://doi.org/10.5281/zenodo.5086370)
[![Python package](https://github.com/tadz-io/hydropt/actions/workflows/python-package.yml/badge.svg)](https://github.com/tadz-io/hydropt/actions/workflows/python-package.yml)
[![Codecov](https://img.shields.io/codecov/c/github/tadz-io/hydropt)](https://app.codecov.io/gh/tadz-io/hydropt)
## Description
<!-- start_ppi_description -->
HYDROPT is an open-source framework for forward and inverse modelling of multi- and hyperspectral observations from oceans, coastal and inland waters. The remote sensing reflectance, R<sub>rs</sub>, is obtained by specifying the inherent optical properties (IOP) of the water column as well as the sensor viewing geometry and solar zenith angle. Our framework is based on radiative transfer principles and is sensor agnostic allowing for R<sub>rs</sub> to be calculated for any wavelength in the 400 - 710 nm range. 



Inversion of R<sub>rs</sub> spectra is achieved by minimizing the difference between the HYDROPT forward calculations and the reflectance measured by the sensor. Different optimization routines can be selected to minimize the cost function. An extensive description of the theoretical basis of the framework as well as applications are provided in the following scientific papers:

> Holtrop, T. & Van Der Woerd, H.J. (*pre-print*) HYDROPT: An open-source framework for fast inverse modelling of multi- and hyperspectral observations from oceans, coastal and inland waters, **2021**. [doi:10.13140/RG.2.2.12314.77768](https://www.researchgate.net/publication/352002441_HYDROPT_An_open-source_framework_for_fast_inverse_modelling_of_multi-_and_hyperspectral_observations_from_oceans_coastal_and_inland_waters?channel=doi&linkId=60b50b3492851cd0d98c7970&showFulltext=true)

>Van Der Woerd, H.J. & Pasterkamp, R. HYDROPT: A fast and flexible method to retrieve chlorophyll-a from multispectral satellite observations of optically complex coastal waters. *Remote Sensing of Environment* **2008**, 112, 1795–1807. [doi:10.1016/j.rse.2007.09.001](https://www.sciencedirect.com/science/article/abs/pii/S003442570700421X?via%3Dihub)

<!-- stop_ppi_description -->
## Features

- Specification of IOP models for forward and inverse modelling
- Sensor agnostic calculations of R<sub>rs</sub> in 400 - 710 nm range
- Calculation of R<sub>rs</sub> at nadir; off-nadir angles will be implemented in the future
- Specification of solar zenith angle will be implemented in the future (30 degrees sza by default)
- Levenberg-Marquardt optimization is used for the inversion; future versions will be able to select the full range of optimization routines offered in ```SciPy``` and ```LMFIT``` libraries


## Installation

...

```bash
pip install ...
```

## Usage
- add binder
```python
import hydropt as hd
```


## Contributing
...

## License
**GNU AFFERO GENERAL PUBLIC LICENSE**
Version 3, 19 November 2007