from setuptools import setup

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

__version__ = 'unknown'
exec(open('ctdna/version.py').read())

setup(
      name='ctdna',                                  # package name
      # packages=find_packages(),
      packages=['ctdna', 'tests'],
      version=__version__,
      description='ctdna computes the expected tumor detection size for a shed biomarker and sampling frequency.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn'],
      url='https://github.com/reiterlab/ctdna',
      author='Johannes Reiter',
      author_email='johannes.reiter@stanford.edu',
      license='GNUv3',
      classifiers=[
        'Programming Language :: Python :: 3.6',
      ],
      test_suite='tests',
      entry_points={
        'console_scripts': [
            'ctdna = ctdna.ctdna:main'
        ]
      }
)
