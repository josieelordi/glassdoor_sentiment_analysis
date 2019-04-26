These libraries need to be installed:

pip install -U spacy
conda install pandas
python3 -m pip install --upgrade pandas



The included files:

**Original Data Set**

employee_review.csv

**X_ hold textual data like summary, pros, cons, etc. While Y_ is the corresponding numerical label for x test**

X_test.csv
X_train.csv
Y_test.csv
Y_train.csv

**Predictions for all the subcategories**

predictions.txt

**Cleans the data by splitting it, removing stopwords, stemming and tokenizing**

preprocess.py 

**Holds the different models we used to test and train the data**

classifier.py


