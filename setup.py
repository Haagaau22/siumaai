from setuptools import setup, find_packages

# import os
# with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf8')as f:
#     require_list = [line.strip() for line in f if line.strip()]
#     print(require_list)


setup(
    name='siumaai',
    version='0.0.1',
    description='a siumaai for nlp',
    license='Apache License 2.0',
    author='Zonzely',
    # install_requires=require_list,
    install_requires=[
        'torch',
        'transformers',
        'pytorch-lightning'
    ],
    packages=find_packages()
)
