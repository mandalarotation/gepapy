from setuptools import setup  # type: ignore

setup(
    name="gen_scheduling_cuda",
    version="1.1.1",
    description="Developing a library with Cuda and Python for solving scheduling problems on GPU",
    author="Jean Carlo Jimenez Giraldo",
    author_email="mandalarotation@gmail.com",
    license="MIT",
    url="https://github.com/mandalarotation/GenSchedulingCuda-GSC",
    package_data={
        "gen_scheduling_cuda": [
            "py.typed",
            "single_machine.pyi",
            "flow_shop.pyi",
            "job_shop.pyi",
            "operations.pyi",
        ],
        "gen_scheduling_cuda.kernels": [
            "py.typed",
            "crossA0001.pyi",
            "fitnessA0001.pyi",
            "fitnessSM0001.pyi",
            "mutationA0001.pyi",
            "permutationA0001.pyi",
        ],
    },
    packages=[
        "gen_scheduling_cuda",
        "gen_scheduling_cuda.kernels",
        "gen_scheduling_cuda.exceptions",
    ],
    scripts=[],
    install_requires=["numpy", "cupy", "numba", "typing"],
)
