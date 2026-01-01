# ====================================== LIBRARIES REQUIRED FOR SETUP ====================================>
from setuptools import find_packages,setup
from typing import List
# ========================================================================================================>


# ============================== FUNCTION TO ACCESS MY REQUIREMENT.TXT FILE ==============================>
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    THIS FUNCTION WILL RETURN THE LIST OF REQUIREMENTS
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
# ========================================================================================================>


# ======================================= MY MAIN CODE FOR SETUP.PY ======================================>
setup(
    name = 'mlprojects',
    version = '0.0.1',
    author = 'Dubey',
    author_email = 'kumarshubhamdubey20@gmail.com',
    packages = find_packages(),
    # FUNCTION IS CALLED HERE ------------------------------------->
    install_requires = get_requirements('requirements.txt')
)
# ========================================================================================================>
