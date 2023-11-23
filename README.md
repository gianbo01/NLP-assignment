# NPL Project
## Classification of Medical/Non-Medical text

There are two python code to run
1. get_files.py
2. classifier.py

### Get Files code
It downloads all the text pages from Wikipedia contained in the category and subcategories, by passing in input the main category using WikipediaAPI.
In my case i decide to download all pages of "Medical_terminology" category

### Classifier code
This code it loads all the files in the Medical and Non-Medical folder.
The code creates two list of all words in the texts, that are _alpha_ and not contained in the _Stop Words List_, using the tokenization functions provided by the NLTK library.
Then it subs the two list of words to find and remove the words that are in both lists.


After that following the **Naive Bayes Classifier** pseudo-code
