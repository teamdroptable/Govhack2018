# Govhack2018

Two parts to the project:

Front End (the html and css file)
Open up the webpage see what it looks like, this is the screen a user would see, they would input their details and the website would connect to a server that would be running the Python code.

Back End (model_creator and processingFile)
The website would run processingFile.py with the inputs from the website to calculate what level of crash is likely to occur (ranging from property damage, medical, fatality).

The processingFile.py in turn uses allFactors.pickle in its processing.  allFactors.picke is a hyper dimensional model generated by model_creator.py which takes the .csv file and scans uses the 54,000 data entries to generate an SVM model.

The .csv file is data taken from https://www.data.act.gov.au/Transport/ACT-Road-Crash-Data/6jn4-m8rx that we've transformed into a mathematical variant for SVM processing.

The current model nearly always generates just property damage as the result but the model is scalable.  We could expand it to include car-models, more exact timings, infrastructure repair costs, demographs of humans involved, etc.  THe setup us incredibly scalable and is designed to literally to using Gigabyes of raw data to optmize predictions.  Not just for what sort of crash but likeihood of crashing at all! 
