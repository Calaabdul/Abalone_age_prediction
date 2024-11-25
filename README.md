# Supervised machine Learning in predicting Abalone's age

###   ABALONE
Predict the age of abalone from physical measurements
Dataset Characteristics
- Tabular
Subject Area
- Biology
Associated Tasks
- Regression
### Instances
4177
### Features
8

### Description
Predicting the age of abalone from physical measurements.  The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task.  Other measurements, which are easier to obtain, are used to predict the age.  Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.

From the original data examples with missing values were removed (the majority having the predicted value missing), and the ranges of the continuous values have been scaled for use with an ANN (by dividing by 200)


Given is the attribute name, attribute type, the measurement unit and a brief description.  The number of rings is the value to predict: either as a continuous value or as a classification problem.

### Features / Data Type / Measurement Unit 
- Sex / M, F, and I (infant)
- Length in mm --> Longest shell measurement
- Diameter in mm --> perpendicular to length
- Height in mm --> with meat in shell
- Whole_weight in grams --> whole abalone
- Shucked_weight in grams --> weight of meat
- Viscera_weight in grams --> gut weight (after bleeding)
- Shell_weight in grams --> after being dried
- Rings --> integer / +1.5 gives the age in years



### License
This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
