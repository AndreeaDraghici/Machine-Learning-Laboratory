import time

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''
  Created by: Draghici Andreea-Maria
  Date: Dec 2022
'''


class Printer(object) :

    @staticmethod
    def display(y_test, y_pred, clf, start_time) :
        print("\nThe " + clf.__class__.__name__ + " took: %s seconds." % (time.time() - start_time))
        print("Estimation of model accuracy with " + clf.__class__.__name__ + " is ",
              accuracy_score(y_test, y_pred) * 100, '%')
        print("\n")

class Parser(Printer) :
    def __init__(self) :
        pass

    @staticmethod
    def parse_method() :
        try :
            mnist = pd.read_csv('./input/mnist_test.xls')
            X, y = mnist.drop('label', axis=1).to_numpy(), mnist['label'].to_numpy()
            X_train, X_test, y_train, y_test = X[:5000], X[5000 :10000], y[:5000], y[5000 :10000]

            log_clf = LogisticRegression()
            rnd_clf = RandomForestClassifier()
            bag_clf = BaggingClassifier(RandomForestClassifier(random_state=1),
                                        3)  # BaggingClassifier with RandomForestClassifier
            xgb_clf = xgb.XGBClassifier(random_state=1, learning_rate=0.01)

            # Estimation of model accuracy
            for clf in (log_clf, rnd_clf, xgb_clf, bag_clf) :
                start_time = time.time()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                Parser.display(y_test, y_pred, clf, start_time)

        except Exception as exception :
            print(f"Filed to run the implementation due to: {exception}\n")
