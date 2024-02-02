from setuptools import setup

setup(
    name='sapienipc',
    version='0.0.0',
    packages=['sapienipc'],
    package_dir={'':'src'},
    install_requires=[
        'numpy',
        'torch',
        'sapien',
        'warp',
        'meshio',
    ],
)
