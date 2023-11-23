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
folder_path = "./non_medical"
med_data = []
data = []
stemmer = PorterStemmer()

def dict_sub(dict1, dict2):
    new_dict = {}
    for word,count in dict1:
        if word not in dict2:
            new_dict[word] = count

    return new_dict

# non med tokenization

if os.path.exists(folder_path):
    print("reading non_medical")
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Open and read the content of each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            data += word_tokenize(content.lower())    
else:
    print(f"Folder '{folder_path}' does not exist.")

result = [elem for elem in data if elem.lower() not in stoplist and elem.isalpha()]


# Create a counter of words
#stemmed_words = [stemmer.stem(word) for word in result]

non_med_sum = len(result)

word_counter = Counter(result)
sorted_elements = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
#non_med_words = {word: {'count': count} for word, count in sorted_elements}


#-----------------------------------------------------------------------------------

# med tokenization
if os.path.exists(med_folder_path):
    print("reading medical")
    # List all files in the folder
    files = [f for f in os.listdir(med_folder_path) if os.path.isfile(os.path.join(med_folder_path, f))]

    # Open and read the content of each file
    for file_name in files:
        file_path = os.path.join(med_folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            med_data += word_tokenize(content.lower())
else:
    print(f"Folder '{med_folder_path}' does not exist.")

med_result = [elem for elem in med_data if elem.lower() not in stoplist and elem.isalpha()]


# print(data)
# Create a counter of words

#stemmed_words = [stemmer.stem(word) for word in med_result]

med_sum = len(med_result)

med_word_counter = Counter(med_result)
med_sorted_elements = sorted(med_word_counter.items(), key=lambda x: x[1], reverse=True)

tmp_med_words = dict_sub(med_sorted_elements,sorted_elements)
tmp_non_med_words = dict_sub(sorted_elements,med_sorted_elements)

med_words = {word: {'freq': (count/med_sum)} for (word, count) in tmp_med_words.items()} # if word not in sorted_elements and count > 9 and count < 2000}
non_med_words = {word: {'freq': (count/non_med_sum)} for (word, count) in tmp_non_med_words.items()} # if word not in med_sorted_elementsand count > 9 and count < 2000}


jsonName = "medword.json"
# save dict in a json file
with open(jsonName, 'w', encoding='utf-8') as jsonFile:
    json.dump(med_words, jsonFile, indent=2)

jsonName = "non_medword.json"
# save dict in a json file
with open(jsonName, 'w', encoding='utf-8') as jsonFile:
    json.dump(non_med_words, jsonFile, indent=2)

print(f"med_words ->{med_sum}")
print(f"non med_words ->{non_med_sum}")

#for (word, freq) in non_med_words.items():
#    try:
#        print(f"{word} -> {freq}")
#    except UnicodeEncodeError:
#        print('error')
    
9
test_label = [0,1,1,0,1,1,0,0,0,1]

for i in range(10):

    tot = 1098 + 900
#    tot = med_sum + non_med_sum
#    med_prob = np.log(med_sum/tot)
#    non_med_prob = np.log(non_med_sum/tot)
    med_prob = -np.log(1098/tot)
    non_med_prob = -np.log(900/tot)

    print(f"file{i}")

    file_path = os.path.join("./test", f"file{i}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    tokenized_text = word_tokenize(content.lower())
    result_file = [elem for elem in tokenized_text if elem not in stoplist and elem.isalpha()]

    #stemmed_words = [stemmer.stem(word) for word in result]
    counter = Counter(result_file)
    sorted_text = sorted(counter.items(), key=lambda x: x[1], reverse=True)


    # calculate medical probability
#    for word, count in sorted_text:
#        if word in med_words:
#            med_prob += (1-np.log(count/(med_words[word]['count'])))
#            #print(np.log(count/med_words[word]['count']))
#            #p_w_c = count / (len(sorted_text))
#            #med_prob += np.log(p_w_c) #(count*(med_words[word]['count']/len(med_words)))
#            
#
#    # calculate NON medical probability
#    for word, count in sorted_text:
#        if word in non_med_words:
#            non_med_prob += (1-np.log(count/(non_med_words[word]['count'])))
#            #p_w_c = count / (len(sorted_text))
#            #non_med_prob += np.log(p_w_c) #(count*(non_med_words[word]['count']/len(non_med_words)))
#        # print(f"{word} - {count} -> {count_non_med}     ({(count/len(non_med_words))})")

            # parola / tot


    for word,count in sorted_text:
        if word in med_words:
            med_prob -= np.log(med_words[word]['freq'])
            #print(med_prob)

    for word,count in sorted_text:
        if word in non_med_words:
            non_med_prob -= np.log(non_med_words[word]['freq'])
    
    print(f"med text probability -> {med_prob}")
    print(f"NON med text probability -> {non_med_prob}")


    if (med_prob) > (non_med_prob):
        print("medical text")
    else:
        print("non medical text")

    if test_label[i] == 1:
        print("file medico")
    else:
        print("file non medico")

    print("--------------------------------------------------------------------")