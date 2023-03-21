#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5

from distutils.core import setup

setup(
    name='add_airush classification',
    version='1.1',
    install_requires=[
        'protobuf==3.18.3',
        'matplotlib==3.1.1',
        'pandas==0.23.4',
        'scikit-learn==0.22',
        'transformers==4.15.0'
    ]
)
