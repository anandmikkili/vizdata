import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='vizdata',  
     version='0.1',
     scripts=['vizdata'] ,
     author="Ananda Rao Mikkili",
     author_email="ananda.rao@6dtech.co.in",
     description="Data Visualization Utility for ML/DL/AI Purposes",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/anandmikkili/vizdata",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: 6D Approved :: 6D Technologies Pvt. Ltd. License",
         "Operating System :: OS Independent",
     ],
 )
