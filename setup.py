from setuptools import setup, find_packages


# requirements list :
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="relapse_prediction",
    version='1.0.0',
    description="Package for predicting relapse in Glioblastoma patients",
    author="EL AICHI MOHAMMED",
    author_email="mohammed.el-aichi@gustaveroussy.fr",
    url="https://github.com/maichi98/RELAPSE_PREDICTION",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
