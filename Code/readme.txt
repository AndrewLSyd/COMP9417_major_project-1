The number at the beginning of the filenames implies when it should be run.
Files with smaller number (I.e. 00_main.py) will be run first.

For convenience of use, 00_main.py will be in charge of running all other scripts. I.e., running the 00_main.py will preprocess the data, extract thee features, train the models and save the results automatically.

Before running 00_main.py, make sure you modify the rscript_location and file_location to where your Rscript program is located and where your scripts are located respectively.

Make the following changes.
rscript_location = "{YourOwnPath}/Rscript.exe"
file_location    = "{YourOwnPathToTheScripts}"

After that, you should be able to run all our scripts at once with a single click. And all temporary results will be displayed and the final results will be saved in the correct folders.
