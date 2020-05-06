from setuptools import setup

with open('README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

__version__ = 'unknown'
exec(open('cbmlb/version.py').read())

setup(
      name='cbmlb',                                  # package name
      # packages=find_packages(),
      packages=['cbmlb', 'tests'],
      version=__version__,
      description='CBMLB computes the expected detection size for a given biomarker and examination schedule.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn'],
      url='https://github.com/johannesreiter/cbmlb',
      author='Johannes Reiter',
      author_email='johannes.reiter@stanford.edu',
      license='GNUv3',
      classifiers=[
        'Programming Language :: Python :: 3.6',
      ],
      test_suite='tests',
      entry_points={
        'console_scripts': [
            'cbmlb = cbmlb.cbmlb:main'
        ]
      }
)
