from setuptools import setup  # type: ignore

with open("README.rst", "r") as readme_file:
    readme = readme_file.read()


setup(
    name="gepapy",
    version="1.1.3",
    description="Developing a library with Cuda and Python for solving scheduling problems on GPU",
    long_description=readme,
    author="Jean Carlo Jimenez Giraldo",
    author_email="mandalarotation@gmail.com",
    license="MIT",
    url="https://github.com/mandalarotation/GenSchedulingCuda-GSC",
    package_data={
        "gepapy": [
            "py.typed",
            "single_machine.pyi",
            "flow_shop.pyi",
            "job_shop.pyi",
            "operations.pyi",
        ],
        "gepapy.kernels": [
            "py.typed",
            "crossA0001.pyi",
            "fitnessA0001.pyi",
            "fitnessSM0001.pyi",
            "mutationA0001.pyi",
            "permutationA0001.pyi",
        ],
    },
    packages=[
        "gepapy",
        "gepapy.kernels",
        "gepapy.exceptions",
    ],
    scripts=[],
    install_requires=["numpy", "cupy", "numba", "typing"],
)
