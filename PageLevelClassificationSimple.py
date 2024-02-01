# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset


# Load the dataset
imdb = load_dataset("imdb")
# print(imdb['train']['text'][0])
# print(imdb['train']['label'][0])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(imdb['train']['text'], imdb['train']['label'], test_size=0.2, random_state=42)
# print(X_train[0])
# print(X_test[0])
# print(y_train[0])
# print(y_test[0])

# # Create the Bag of Words vectorizer
# vectorizer = CountVectorizer()
# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the training data
X_train_bow = vectorizer.fit_transform(X_train)

# Create the classifier
clf = MultinomialNB()
# # Create the SVM classifier
# clf = SVC()
# # Create the Random Forest classifier
# clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train_bow, y_train)

# Transform the test data
X_test_bow = vectorizer.transform(X_test)

# Predict the labels
predicted = clf.predict(X_test_bow)

# Print the classification report
print(classification_report(y_test, predicted))