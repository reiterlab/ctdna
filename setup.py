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
      description='ctdna computes the expected tumor detection size at a given sampling frequency.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib'],
      setup_requires=['pytest-runner', 'flake8'],
      tests_require=['pytest', 'pytest-cov'],
      extras_require={'plotting': ['matplotlib', 'seaborn', 'jupyter']},
      url='https://github.com/reiterlab/ctdna',
      author='Johannes Reiter',
      author_email='johannes.reiter@stanford.edu',
      license='GNUv3',
      keywords=['ctdna', 'cancer', 'cancer early detection', 'cancer screening', 'treatment monitoring'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
      ],
      test_suite='tests',
      entry_points={
        'console_scripts': [
            'ctdna = ctdna.ctdna:main'
        ]
      }
)
