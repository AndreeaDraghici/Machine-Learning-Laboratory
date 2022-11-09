import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB


class Printer(object) :

    # comparing actual response values (y_test) with predicted response values (y_pred)
    @staticmethod
    def display(y_prediction, y_test) :
        print("\nAccuracy : ", accuracy_score(y_test, y_prediction) * 100)
        print("\nReport : \n", classification_report(y_test, y_prediction))


class Parser(Printer) :

    def __init__(self) :
        pass

    def parse_method(self) :
        try :
            data = pd.read_csv("./input/SPAM text message 20170820 - Data.csv")  # load the csv

            y = CountVectorizer()
            X = y.fit_transform(data['Message']).toarray()  # store the feature matrix (X) and response vector (y)
            X_train, X_test, y_train, y_test = train_test_split(X, data.Category, test_size=0.2,
                                                                random_state=32)  # splitting X and y into training and testing sets

            X_train.shape, X_test.shape, y_train.shape, y_test.shape

            self.using_multinomial_naive_bayes(X_test, X_train, y_test, y_train)
            self.using_gaussian_naive_bayes(X_test, X_train, y_test, y_train)
            self.using_decision_trees(X_test, X_train, y_test, y_train)

        except Exception as exception :
            print(f"Filed to run the implementation due to: {exception}\n")

    @staticmethod
    def using_decision_trees(X_test, X_train, y_test, y_train) :

        decision_tree = DecisionTreeClassifier().fit(X_train, y_train)  # training the model on training set
        prediction_decision_tree = decision_tree.predict(X_test)  # making predictions on the testing set

        print('\n-----Decision Tree score is------')
        Parser.display(prediction_decision_tree, y_test)

    @staticmethod
    def using_multinomial_naive_bayes(X_test, X_train, y_test, y_train) :
        naive_bayes = MultinomialNB().fit(X_train, y_train)  # training the model on training set
        prediction_naive_bayes = naive_bayes.predict(X_test)  # making predictions on the testing set

        print('\n-----Multinomial Naive Bayes score is------')
        Parser.display(prediction_naive_bayes, y_test)

    @staticmethod
    def using_gaussian_naive_bayes(X_test, X_train, y_test, y_train) :
        naive_bayes = GaussianNB().fit(X_train, y_train)  # training the model on training set
        prediction_naive_bayes = naive_bayes.predict(X_test)  # making predictions on the testing set

        print('\n-----Gaussian Naive Bayes score is------')
        Parser.display(prediction_naive_bayes, y_test)
