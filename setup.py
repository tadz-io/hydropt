from setuptools import setup
import versioneer

def get_long_description():
    """Extract description from README.md, for PyPI's usage"""
    def process_ignore_tags(buffer):
        return "\n".join(
            x for x in buffer.split("\n") if "<!-- ignore_ppi -->" not in x
        )
    try:
        fpath = "README.md"
        with open(fpath, encoding="utf-8") as f:
            readme = f.read()
            desc = readme.partition("<!-- start_ppi_description -->")[2]
            desc = desc.partition("<!-- stop_ppi_description -->")[0]
            return process_ignore_tags(desc.strip())
    except IOError:
        return None

setup(name='hydropt-oc',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      url='https://github.com/tadz-io/hydropt',
      author='Tadzio Holtrop',
      author_email='tadzio.holtrop@icloud.com',
      description='''HYDROPT: a Python Framework for Fast Inverse Modelling of
       Multi- and Hyperspectral Ocean Color Data''',
      long_description=get_long_description(),
      long_description_content_type="text/markdown",
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