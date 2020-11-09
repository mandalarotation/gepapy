from setuptools import setup

setup(
    name="gsc",
    version="1.1.1",
    description="Developing a library with Cuda and Python for solve Scheduling problems on GPU",
    author="Jean Carlo Jimenez Giraldo",
    author_email="mandalarotation@gmail.com",
    url="https://github.com/mandalarotation/GenSchedulingCuda-GSC",
    packages=["gsc", "gsc.kernels"],
    scripts=[],
    install_requires=["numpy", "cupy", "numba", "typing"],
)
