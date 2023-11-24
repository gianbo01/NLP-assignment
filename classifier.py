import nltk
import os
import numpy as np
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

stoplist = stopwords.words('english')
med_folder_path = "./medical"
non_med_folder_path = "./non_medical"

stemmer = PorterStemmer()


# sub 2 dictionary
def dict_sub(dict1, dict2):
    new_dict = {}
    for (word, count) in dict1.items():
        if word not in dict2:
            new_dict[word] = count
    
    # return a new dictionary
    return new_dict



# files tokenization
def docs_to_token(folder_path):
    stoplist = stopwords.words('english')
    data = []

    if os.path.exists(folder_path):
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # Open and read the content of each file
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                #tokenization
                data += word_tokenize(content.lower())    
    else:
        print(f"Folder '{folder_path}' does not exist.")

    # remove stop words
    result = [elem for elem in data if elem.lower() not in stoplist and elem.isalpha()]
    return result

# return the number of documents in a folder
def count_files(folder_path):
    if os.path.exists(folder_path):
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return len(files)

# passing the folder path return the tokenized list
med_result  = docs_to_token(med_folder_path)
non_med_result = docs_to_token(non_med_folder_path)

# stemming
med_stemmed_words = [stemmer.stem(word) for word in med_result]
non_med_stemmed_words = [stemmer.stem(word) for word in non_med_result]

# counting occurence of every word
med_word_counter = Counter(med_stemmed_words)
non_med_word_counter = Counter(non_med_stemmed_words)

# remove the words in both dictionary
tmp_med_words = dict_sub(med_word_counter,non_med_word_counter)
tmp_non_med_words = dict_sub(non_med_word_counter,med_word_counter)

# number of words in the bag of words
med_sum = len(tmp_med_words)
non_med_sum = len(tmp_non_med_words)

# calculate probability following Naive Bayse rules
med_words = {word: {'freq': (count/med_sum)} for (word, count) in tmp_med_words.items()}
non_med_words = {word: {'freq': (count/non_med_sum)} for (word, count) in tmp_non_med_words.items()}


med_file = count_files(med_folder_path)     # number of medical file in the folder
non_med_file = count_files(non_med_folder_path)  # number of NON medical file in the folder

test_label = [0,1,1,0,1,1,0,0,0,1]

# initialization of variables
tp = 0      # True Positive
tn = 0      # True Negative
fp = 0      # False Positive
fn = 0      # False Negative

for i in range(10):

    # calculate the sum of total docs
    tot = med_file + non_med_file

    # calculate the p(c_j)
    med_prob = -np.log(med_file/tot)
    non_med_prob = -np.log(non_med_file/tot)

    print(f"file{i}.txt")

    file_path = os.path.join("./test", f"file{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # tokenization of the input file
    tokenized_text = word_tokenize(content.lower())
    result_file = [elem for elem in tokenized_text if elem not in stoplist and elem.isalpha()]

    # stemming of the input file
    stemmed_words = [stemmer.stem(word) for word in result_file]

    # counting the words occurrences
    counter = Counter(stemmed_words)

    # sorting the words occurrences but its irrilevant for the classification
    sorted_text = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # iterate every word to calculate the medical probability
    for word,count in sorted_text:
        # if it is in the medical bag of words
        if word in med_words:
            # sub the log frequencies because a log of decimal number is negative so i have positive results
            med_prob -= np.log(med_words[word]['freq'])

    # iterate every word to calculate the NON medical probability
    for word,count in sorted_text:
        # if it is in the NON medical bag of words
        if word in non_med_words:
            # sub the log frequencies because a log of decimal number is negative so i have positive results
            non_med_prob -= np.log(non_med_words[word]['freq'])
    
    print(f"med text probability -> {med_prob}")
    print(f"NON med text probability -> {non_med_prob}")

    if (med_prob) > (non_med_prob):
        print("Result -> medical text")
        if test_label[i] == 1:
            print("Label -> medical file")
            tp += 1
        else:
            print("Label -> non medical file")
            fp += 1
    else:
        print("Result -> non medical text")
        if test_label[i] == 1:
            print("Label -> medical file")
            fn += 1
        else:
            print("Label -> non medical file")
            tn += 1

    print("--------------------------------------------------------------------")

# calculate accuracy
accuracy = (tp + tn) / (tp + fp + tn + fn)
print(f"Accuracy -> {accuracy*100}%")

# calculate precision
precision = tp / (tp+fp)
print(f"Precision -> {precision*100}%")

# calculate recall
recall = tp / (tp + fn)
print(f"Recall -> {recall*100}%")