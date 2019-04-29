from flask import Flask,render_template,request,url_for,jsonify

#EDA Packages
import pyodbc
import pandas as pd
from sklearn import metrics

# ML Packages
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)


@app.route("/train/", methods=['POST'])
def train():
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=fyp2019.database.windows.net;'
                      'Database=fyp;'
                      'UID=test;PWD=login@123') 
    cursor = conn.cursor()
    sql = "exec p_getAllDataWithVType"
    df = pd.read_sql(sql, conn)
    X = df.drop(['vehicle_type'], axis=1)
    y = df['vehicle_type'].values
    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 3)
    # Fit the classifier to the data
    knn.fit(X_train,y_train)
    knn.score(X_test, y_test)
    pred = knn.predict(X_test)

    prediction = metrics.accuracy_score(y_test, pred)
    pred_list = prediction.tolist()
    return jsonify(pred_list)


@app.route("/predict/",methods=['POST'])
def predict():
    data = [request.get_json()]
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=fyp2019.database.windows.net;'
                      'Database=fyp;'
                      'UID=test;PWD=login@123') 
    cursor = conn.cursor()
    sql = "exec p_getAllDataWithVType"
    df = pd.read_sql(sql, conn)
    X = df.drop(['vehicle_type'], axis=1)
    y = df['vehicle_type'].values
    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)

    prediction = (knn.predict(data))
    pred_list = prediction.tolist()
    return jsonify(pred_list)



if __name__ == '__main__':
	app.run()