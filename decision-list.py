#***************************************************WORD SENSE DISAMBIGUATION USING DECISION LIST CLASSIFIER********************************************#
#Date: 27th October, 2020
#Authors: Team 9 - AIT 590-001 - Asmita Singh, Amrita Jose, Prateek Chitpur

#What is the program about?
#Word sense disambiguation refers to a technique in natural language processing that involves identifying the meaning or sense of the word in a context.
#This program aims to implement word sense disambiguation using decision list classifiers. Decision list classifiers are popular choices to resolve lexical
#ambiguities. This program uses decision lists to identify and differentiate collocationally distributed words for each sense of the ambiguous word,
#with respect to their log likelihood. The program reads a train file - 'line-train.xml' (comprising of training data) and a test file - 'line-test.xml'
#(comprising of test data) in order to train a decision list classifier model and apply the model on the test data to measure its efficiency and
#predictive capabilities. The line-train.xml consists of words with the ambiguous word - 'line', used in either of the two senses - phone or product.
#The line-test.xml consists of the ambiguous word - "line" without the senses. The program tries to learn and generate a model by identifying features from
#the traiing data and compute the log likelihood of each sense - phone and product. 

#What are the features used to train the model?
#The features learned from the train set include identifying a bag of words(collocations) for each sense - product and phone. In this program, n previous
#and n future words were identified for each sense and their individual frequency distributions were computed in order to calculate the combined frequency
#for each sense. The bag of words are generated up till 7 previous and 7 future words as 7 words give the best accuracy for the model built
#in the program. The conditional probabilities calculated for the bag of words were then applied to the test set to determine the sense of context. 

#Program Steps/Algorithm Execution
#1. Read the train and test set from command line
#2. Create a function to parse the training and test data and remove stopwords and punctuation as part of data preperocessing.
#3. Create Bag of Words for each sense by selecting upto 7 previous and 7 future words and locate their indices
#4. Compute frequency distributions for the bag of words.
#5. Calculate Condition probability distribution for the bag of words using logarithm of ratio of sense probabilities. This gives us the log likelihood
#   of each sense.
#6. Calculate combined frequencies for each sense and determine the majority sense in training data.
#7. Apply the calculated conditional probabilities for the bag of words to the test set to predict the sense of the context.
#8. Write the decision list rules to a file called my-decision-list.txt.
#9. Output the answer tags created for each sentence to STDOUT. The results will be displayed in my-line-answers.txt when run from console.

#Components for Execution
#1. line-train.xml : Train data with examples of the word line used in the sense of a phone line and a product line
#2. line-test.xml : Test data with examples of sentences that use the word line without any sense being indicated
#3. my-decision-list.txt: File into which the decision rules are written.
#   Sample output for my-decision-list.txt

#   ['-1_word_telephone', 8.45532722030456, 'phone']
#   ['-1_word_access', 7.238404739325079, 'phone']
#   ['-1_word_car', -6.507794640198696, 'product']
#   ['-1_word_end', 6.339850002884625, 'phone']
#   ['1_word_dead', 5.930737337562887, 'phone']
#4. my-line-answers.txt: File into which the answer tags for each sentence are output to STDOUT.
#   Sample output for my-line-answers.txt

#   <answer instance="line-n.w8_059:8174:" senseid="phone"/>
#   <answer instance="line-n.w7_098:12684:" senseid="phone"/>
#   <answer instance="line-n.w8_106:13309:" senseid="phone"/>
#   <answer instance="line-n.w9_40:10187:" senseid="phone"/>
#   <answer instance="line-n.w9_16:217:" senseid="product"/>

#5. line-answers.txt: Gold standard key data which will be compared with my-line-answers.txt to determine the accuracy of the classifier.
#6. scorer.py : Utility program which will take as input sense tagged output and compare it with the gold standard "key" data in line-answers.txt

#Instructions for Execution
#1. The classifier should run from the command line as follows: -

#$ python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt

#2. The scorer file should be run as follows: -

#$ python scorer.py my-line-answers.txt line-answers.txt

#Performance Evaluation: Accuracy and Confusion Matrix of the Generated Classifier

#   Actual Accuracy: 57.14%
#   Accuracy obtained from trained model: 83.33%
#   Confusion matrix:
#   col_0    phone  product
#   row_0                  
#   phone       62       11
#   product     10       43

#The baseline accuracy is 57.14% while the accuracy of the implemented model is 83.33%

#References Used
#1. Dr. Liao, D. (2020, Aug). Code Examples for NLP Basic Text Processing
#Retrieved October 21, 2020, from https://mymasonportal.gmu.edu/bbcswebdav/pid-11302947-dt-content-rid-187517083_1/courses/81555.202070/Liao_NLP_text_processing.html
#2. Python BeautifulSoup XML Parsing (2011, November). Stackoverflow.
#Retrieved October 22, 2020, from https://stackoverflow.com/questions/4071696/python-beautifulsoup-xml-parsing
#3. Find out word at specific index (2016, November). Stackoverflow.
#Retrieved October 22, 2020, from https://stackoverflow.com/questions/40675958/find-out-word-at-specific-index
#4. Kallio, J. (2014, December). YarowskyWSD
#Retrieved October 23, 2020, from  https://github.com/juhokallio/YarowskyWSD/blob/master/yarowsky.py
#5. New to nltk, having trouble with conditional frequency. (2018, June). Stackoverflow.
#Retrieved October 22, 2020, from https://stackoverflow.com/questions/32676319/new-to-nltk-having-trouble-with-conditional-frequency
#6. Hamizi, I. (2019, Mar). NLP-WSD
#Retrieved October 22, 2020, from https://github.com/ikram-hamizi/NLP-WSD/blob/master/decision-list.pl
#7. Madduri, S. (2019, June). Wordsense-Disambiguation
#Retrieved October 22, 2020, from https://github.com/sainikhithamadduri/Wordsense-Disambiguation-/blob/master/decision-list.py
#8. Rapaka, M. (2019, Jan). Word-Sense-Disambiguation
#Retrieved October 23, 2020, from https://github.com/meenarapaka/Word-Sense-Disambiguation/blob/master/decision-list.py
#9. Yarowsky, David. "A DECISION LISTS FOR LEXICAL AMBIGUITY RESOLUTION Application to Accent Restoration in Spanish and French", 
#Department of Computer and Information Science, University of Pennsylvania, Philadelphia PA 19104

#Import libraries
import sys
import re
import bs4
import math
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LidstoneProbDist


#Initialize system arguments 
train_data = sys.argv[1]
test_data = sys.argv[2]
my_decision_list = sys.argv[3]

#Preprocess the data
def data_preprocessing(text_input):
    text_input = text_input.lower()
    text_input = text_input.replace("lines", "line")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text_input)
    text_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    return text_stopwords

#Parse Training Data
train_set = []
trainfile = open(train_data)
data=trainfile.read()
fileparser = BeautifulSoup(data, 'html.parser')
for instance in fileparser.find_all('instance'):
    x = ""
    xmlinstance = dict()
    xmlinstance['id'] = instance['id']
    xmlinstance['sense'] = instance.answer['senseid']
    for word in instance.find_all('s'):
        x = x+ " "+ word.get_text()
    xmlinstance['text'] = data_preprocessing(x)
    train_set.append(xmlinstance)

#Function to create conditional frequency distributions after retrieving a word at a certain index
def create_cond(condfreqdist,data,num):
    word = "line"
    for values in data:
        sense, context = values['sense'], values['text']
        word_index = context.index(word)
        word_index_num = word_index + num
        if len(context) > word_index_num and word_index_num >= 0:
            word_num = context[word_index_num]
        else:
            word_num = ""
        if word_num != '':
            strings = str(num) + "_stringlist_" + word_num
            condfreqdist[strings][sense] += 1
    return condfreqdist
#Call to compute conditional frequency distribution for bag of words
condfreqdist = ConditionalFreqDist()
for i in range(1,7):
    condfreqdist = create_cond(condfreqdist, train_set, i)
    condfreqdist = create_cond(condfreqdist, train_set, -i)

# Calculate Condition probability distribution for bag of words and determining log likelihood of senses
dlist = []
CondProbDist = ConditionalProbDist(condfreqdist, LidstoneProbDist, 0.1)
for strings in CondProbDist.conditions():
    phoneprob = CondProbDist[strings].prob("phone")
    productprob = CondProbDist[strings].prob("product")
    result = phoneprob / productprob
    if result == 0:
        result_lkelhd = 0
    else:
        result_lkelhd = math.log(result, 2)   
    dlist.append([strings, result_lkelhd, "phone" if result_lkelhd > 0 else "product"])
    dlist.sort(key=lambda strings: math.fabs(strings[1]), reverse=True)

#Calculating the combined frequency for each sense and determining sense with the highest frequency in the train data
freq_1=0
freq_2=0
for data_sense in train_set:
    if data_sense['sense'] == "phone":
        freq_1 += 1
    else:
        freq_2 += 1
if freq_1 > freq_2:
    highest_freq = "phone"
else:
    highest_freq = "product"

#Parse Testing Data
test_set = []
testfile = open(test_data)
data=testfile.read()
fileparser = BeautifulSoup(data, 'html.parser')
for instance in fileparser('instance'):
    x = ""
    xmlinstance = dict()
    xmlinstance['id'] = instance['id']
    for word in instance.find_all('s'):
        x = x+ " "+ word.get_text()
    xmlinstance['text'] = data_preprocessing(x)
    test_set.append(xmlinstance)

#Check whether the rule is in alignment with the context
def condition(context, lists):
    list_x,list_y,list_z = lists.split("_")
    list_x = int(list_x)
    ambstring= "line"
    ind = context.index(ambstring)
    wordpos = ind + list_x
    if len(context) > wordpos and wordpos >= 0:
        word_num = context[wordpos]
    else:
        word_num = ""
    return word_num == list_z

#Predicting on test data
predstrings = []
def predict_sense(context,label):
    for strings in dlist:
         if condition(context, strings[0]):
            if strings[1] > 0:
                return ("phone", context, strings[0])
            elif strings[1] < 0:
                return ("product", context, strings[0])
    return (label, context, "default")
for string in test_set:
    stringid = string['id']
    i, _, j = predict_sense(string['text'],highest_freq)
    predstrings.append(f'<answer instance="{stringid}" senseid="{i}"/>')
    print(f'<answer instance="{stringid}" senseid="{i}"/>')

#Writing generated predictions to file
with open (my_decision_list, 'w') as output:
    for string in dlist:
        output.write('%s\n' % string)
    

