# NPL Project
## Classification of Medical/Non-Medical text

There are two python code to run
1. get_files.py
2. classifier.py

### Get Files code
It downloads all the text pages from Wikipedia contained in the category and subcategories, by passing in input the main category using WikipediaAPI.
In my case i decide to download all pages of "Medical_terminology" category for the creation of medical bag-of-words and "History", "Sports" and "Geography" for the non medical bag-of-word

### Classifier code
This code it loads all the files in the Medical and Non-Medical folder.
The code creates two list of all words in the texts, that are _alpha_ and not contained in the _Stop Words List_, using the tokenization functions provided by the NLTK library.
Another process that it do is the Stemming: it takes only the root of the words, so there are more precision and it don't  need to have a lot of words with the same meaning
Then it subs the two list of words to find and remove the words that are in both lists.


After the creation of medical and non-medical bag of word, it follows the **Naive Bayes Classifier** pseudo-code


$$c_{NB} = \underset{c_j \in C}{\mathrm{argmax}} \left[ \log{P(c_j)} + \sum_{i \in positions} \log{P\left(x_i \middle| c_j \right)} \right]$$

where 

number of documents in the class among the number of total documents:

$$ P(c_j) = \frac{N_{c_j}}{N_{total}} $$ 

fraction of the occurences of i-word among all words in the bag of words of the i-class:

$$ P\left( w_i \middle| c_j \right) = \frac{count(w_i, c_j)}{\sum_{w \in V} count(w, c_j)}$$ 

then it assignes the class with the maximum logaritmical sum to the input text
