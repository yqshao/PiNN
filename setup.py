import os
from setuptools import setup, find_packages

setup(name='pinn',
      version='dev',
      description='Pair interaction Neural Network',
      url='https://github.com/yqshao/pinn',
      author='Yunqi Shao',
      author_email='yunqi_shao@yahoo.com',
      license='BSD',
      packages=find_packages(),
      install_requires=['numpy>1.3.0',
                        'ase>=3.19.0',
                        'pyyaml>=3.01'],
      python_requires='>=3.6',
      extras_require={'cpu': ['tensorflow-cpu>=2.1'],
                      'gpu': ['tensorflow>=2.1']},
      entry_points={'console_scripts':
                    ['pinn_train=pinn.trainer:main']}
)
