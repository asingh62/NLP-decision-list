##Date: 27th October, 2020
#Authors: Team 9 - AIT 590-001 - Asmita Singh, Amrita Jose, Prateek Chitpur

# Scorer.py is a utility program which takes sense tagged output file and gold standard key data file as input. It compares the senseid 
# with respect to specific instances of both the files and prints overall accuracy of the tagging and confusion matrix.

#Components for Execution
#1. my-line-answers.tx : Sense tagged output file obtained from the learned features.
#2. line-answers.txt : GOld standard key data file used to compare with the sense tagged output file.

# Sample output

# Actual accuracy: 57.14285714285714%
# Accuracy obtained from trained model: 83.33333333333334%
# Confusion matrix: col_0    phone  product
# row_0                  
# phone       62       11
# product     10       43%

# References:
# 1.Example of Confusion Matrix in Python. Retrieved October 22, 2020, from https://datatofish.com/confusion-matrix-python/

import pandas as pd
import re
import sys

# The sense tagged output file and gold standard key data file passed as arguments in the command line are stored in the variables
arg = sys.argv[0:]
predicted_file = arg[1]
gold_file = arg[2]

# The below function fetches instances and respective senseids 
def obtain_senseIDs(tag_list):
    instances = []
    senseids = {}
    for tag in tag_list:
        find = re.search('<answer instance="(.*)" senseid="(.*)"/>', tag, re.IGNORECASE)
        instance = find.group(1)
        instances.append(instance)
        senseid = find.group(2)
        senseids[instance] = senseid
    return senseids, instances

# The tagged output file is stripped out based on new line
with open(predicted_file, 'r') as tagged_data:
    tagged_list = [tag.rstrip('\n') for tag in tagged_data]
pred_senseids, _ = obtain_senseIDs(tagged_list)                               

# The predicted senseid values of tagged output file are stored in a list
pred_senseids_list = []
for senseid in pred_senseids:
    pred_senseids_list.append(pred_senseids[senseid])

# The provided gold standard data file is stripped out based on new line
with open(gold_file, 'r') as gold_data:
    gold_list = [tag.rstrip('\n') for tag in gold_data]
senseids, instances = obtain_senseIDs(gold_list)

# The senseid values of gold standard data file are stored in a list
senseids_list = []
for senseid in senseids:
    senseids_list.append(senseids[senseid])

# For each instance of tagged output file and gold standard data file, check if the senseids match and count the total matches
match = 0
for instance in instances:
    if(senseids[instance] == pred_senseids[instance]):
        match += 1
match

# The most frequent senseid is obtained from the gold standard data file and calculate the accuracy of it within the same file
instance_size = len(instances)
actual_match_count = 0
for instance in instances:
    if(senseids[instance] == 'phone'):
        actual_match_count += 1

actual_accuracy = (float(actual_match_count) / float(instance_size)) * 100
print("Actual accuracy: "+ str(actual_accuracy) + "%")

# The accuaracy of the learned feature is calculated
model_accuracy = (float(match) / float(instance_size)) * 100
print("Accuracy obtained from trained model: " + str(model_accuracy) + "%")

# Dataframes for lists of senseids of tagged output file and gold standard data file are created
pred_senseid_df = pd.Series((senseid for senseid in pred_senseids_list))
senseid_df = pd.Series((senseid for senseid in senseids_list))

# The below syntax generates and prints the confusion matrix
confusion_matrix = pd.crosstab(pred_senseid_df, senseid_df)
print("Confusion matrix: " + str(confusion_matrix) + "%")


