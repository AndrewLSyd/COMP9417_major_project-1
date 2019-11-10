# Data set up:
1) Download the assignment (https://www.dropbox.com/s/nrm5s3m9f46uves/StudentLife_Dataset.zip?dl=0)
and extract the data
2) Place the StudentLife_Dataset folder (that contains Inputs, Outputs, and
StudnetLife_AssessingMentalHealth.pdf) the same directory as the comp9417_major_project.Rproj file

# Prediction task
## Target
### Flourishing Scale
*"You can decide to predict either the pre-experiment scores or post-experiment scores or even the average
score from pre and post! You just have to have a clear explanation of what you did in your report!"*
~ Gelarah in forum post

* COMP9417_major_project/StudentLife_Datase/Outputs/FlourishingScale.csv
* flourishing scale is calculated as the sum of all 8 questionarre items.
    * range [8, 56]
* the spec says we are meant to perform **both** regression and classification
    * regression: predict post study score [8, 56]
    * classification: split up the scores into a "Low" and "High" category based on a threshold (e.g.
    the median)

### PANAS score

* COMP9417_major_project/StudentLife_Datase/Outputs/panas.csv
* two targets:
    * Positive Affect Score, calculated as the sum of scores on items 1, 3, 5, 9, 10, 12, 14, 16, 17, 19
        * range [10, 50]
        * mean 33.3
        * SD 7.2
    * Negative Affect Score, calculated as the sum of scores on items 2, 4, 6, 7, 8, 11, 13, 15, 18, 20
        * range [10, 50]
        * mean 17.4
        * SD 6.2
* the spec says we are meant to perform **both** regression and classification
    * regression: predict post study positive/negative affect score [10, 50]
    * classification: split up the positive/negative affect scores into a "Low" and "High" category based on
    a threshold (e.g. the median)
    
# Coding standards
Should we agree on conventions we should stick to?

## Case

### Examples
**snake case** - this_var, var_1  
**camel case** - ThisVar, Var1  
**upper case** - THIS_VAR, VAR_1  

### Convention
**constants** - upper?  
**variables** - snake case  
**functions** - snake case  
**classes** - camel case  ?

## Margins
Should we agree on the maximum width of our codes? 80? 100? 120?


