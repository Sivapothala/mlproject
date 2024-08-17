### responsible for creating machine learning models as package.
from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    requiments = []
    Minus_Hypen = '-e .'
    with open(file_path) as f:
        requi= f.read().splitlines()
        requiments= [req.replace("\n","") for req in requi]

        if Minus_Hypen in requiments:
            requiments.remove(Minus_Hypen)
    return requiments
setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Krish',
    author_email= 'pothalasiva26@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    # install_requires = ['pandas','numpy','seaborn']
)