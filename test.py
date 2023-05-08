import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 

le = preprocessing.LabelEncoder()
dataset = pd.read_csv('datasets/kdd_train.csv')
label_names = dataset.labels.unique()
dataset['labels'] = le.fit_transform(dataset['labels'])
cols = dataset.shape[1]
cols = cols - 1
X = dataset.values[:, 0:cols] 
Y = dataset.values[:, cols]
Y = Y.astype('int')

doc = []
for i in range(len(X)):
    strs = ''
    for j in range(len(X[i])):
        strs+=str(X[i,j])+" "
    doc.append(strs.strip())
 

feature_extraction = TfidfVectorizer()
tfidf = feature_extraction.fit_transform(doc)

X = tfidf.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
rfc = svm.SVC(C=2.0,gamma='scale',kernel = 'linear', random_state = 0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test) 
svm_acc = accuracy_score(y_test,y_pred)*100
print("SVM Accuracy : "+str(svm_acc)+"\n")



