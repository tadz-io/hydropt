from setuptools import setup
import versioneer

setup(name='hydropt',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='HYDROPT',
      url='https://github.com/tadz-io/hydropt',
      author='Tadzio Holtrop',
      author_email='tadzio.holtrop@icloud.com',
      license='AGPL-3.0',
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