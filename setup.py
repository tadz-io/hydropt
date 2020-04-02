from setuptools import setup

setup(name='hydropt',
      version='0.2.0',
      description='HYDROPT',
      url='',
      author='',
      author_email='',
      license='',
      packages=['hydropt'],
      package_dir={'hydropt':'hydropt'},
      package_data={'hydropt': ['data/*.csv']},
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scikit-learn'],
      zip_safe=False)