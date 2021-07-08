from setuptools import setup

setup(name='hydropt',
      version='0.2',
      description='HYDROPT',
      url='',
      author='Tadzio Holtrop',
      author_email='',
      license='GNU Affero General Public License v3.0',
      packages=['hydropt'],
      package_dir={'hydropt': 'hydropt'},
      package_data={'hydropt': ['data/*.csv']},
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scikit-learn',
          'xarray'],
      zip_safe=False)