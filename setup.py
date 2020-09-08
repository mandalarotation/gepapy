from setuptools import setup
setup(
        name="gsc",
        version="1.1.1",
        description="Backend analitica de datos SGC",
        author="Jean Carlo Jimenez Giraldo",
        author_email="mandalarotation@gmail.com",
        url="https://github.com/mandalarotation/GenSchedulingCuda-GSC",
        packages=[
                "gsc",
                "gsc.kernels"],
        scripts=[],
        install_requires=["numpy",
                          "cupy",
                          "numba"])