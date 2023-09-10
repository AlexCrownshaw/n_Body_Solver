from setuptools import setup

setup(
    name='n_body_solver',
    version='0.1.0',
    description='A DNS n body solver using newtons law of gravity',
    url='https://github.com/AlexCrownshaw/n_Body_Solver',
    author='Alex Crownshaw',
    author_email='alex.crownshaw12@gmail.com',
    packages=['n_body_solver'],
    install_requires=["matplotlib==3.5.1", "numpy==1.23.4", "pandas==1.5.1"]
)
