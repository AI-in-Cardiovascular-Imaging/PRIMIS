from setuptools import setup, find_packages

setup(
    name='primis',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "Pillow"
        "python_dateutil",
        "scikit_image",
        "scikit-learn",
        "pandas",
        "tensorboard",
        "matplotlib"
    ],
    extras_require={
        "interactive": [
            "jupyter",
            "jupyterlab",
            "ipykernel"
        ],
        "testing": [
            "torchinfo"
        ]
    }
)
