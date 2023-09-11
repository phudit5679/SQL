#!/usr/bin/env python
# coding: utf-8

# # Required packages

# In[51]:


get_ipython().system('pip install imbalanced-learn')


# In[8]:


pip install scikit-learn==1.2.2


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, classification_report
from imblearn.over_sampling import ADASYN


# # Import dataset

# In[2]:


data = pd.read_csv("D:\หางาน\Churn_Modelling.csv")
data.head()


# # EDA

# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.duplicated().sum()


# In[6]:


data.isnull().sum()


# In[7]:


data.info()


# In[9]:


#data = data.astype({"NumOfProducts":"object","HasCrCard": "object","IsActiveMember":"object"})
#data.info()


# In[8]:


data = data.drop(labels=['RowNumber', 'CustomerId', 'Surname'], axis=1)


# In[9]:


data.describe()


# In[10]:


data["Exited"].value_counts()


# # Visualization

# In[11]:


plt.figure(figsize=(8,6))
ax = sns.countplot(x='Exited', data=data)
for container in ax.containers:
        ax.bar_label(container)

plt.show()


# In[18]:


con_feas = data[["CreditScore","Age","Balance","EstimatedSalary"]]
for feature in con_feas:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=data, x='Exited', y=feature)
    
    plt.title(feature,fontsize=15, weight='bold')
    plt.show


# In[19]:


cate_feas = data[['Geography', 'Gender', 'NumOfProducts', 'Tenure', 'HasCrCard', 'IsActiveMember']]
for feature in cate_feas:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = sns.countplot(data=data, x=feature, hue='Exited', ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container)
    axes[0].set_title(f'{feature}',weight = 'bold',fontsize = 15)

    
    size = data[feature][data['Exited'] == 1].value_counts().values.tolist()
    label = data[feature][data['Exited'] == 1].value_counts().index
    pal = sns.color_palette("YlGnBu", len(data.groupby(feature).size()))
    axes[1].pie(size, labels=label, colors = pal, autopct='%1.1f%%', textprops={'fontsize': 14},
                explode=[0.02 for _ in range(len(label))])
    axes[1].set_title(f'{feature} proportions of people exited',weight = 'bold',fontsize = 15)
    
    plt.legend(bbox_to_anchor=(1, 1) , prop={'size': 15})
    plt.tight_layout()
    plt.show()
    



# In[20]:


sns.heatmap(data = data.corr() , cmap = 'rocket' , annot = True , fmt = '.2f')
plt.show()


# # One Hot Encoding

# In[21]:


cate_feas = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
data_onehot = pd.get_dummies(data, columns = cate_feas)

data_onehot


# In[15]:


cate_feas = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
data_onehot = pd.get_dummies(data, columns = cate_feas)

data_onehot


# In[22]:


data_onehot.rename(columns = {"Geography_France":"France","Geography_Germany":"Germany","Geography_Spain":"Spain",
                            "Gender_Female":"Female","Gender_Male":"Male",
                              "NumOfProducts_1":"1_Product","NumOfProducts_2":"2_Product",
                              "NumOfProducts_3":"3_Product","NumOfProducts_4":"4_Product",
                            "HasCrCard_0":"HasCrCard_No","HasCrCard_1":"HasCrCard_Yes",
                            "IsActiveMember_1":"IsActiveMember_Yes","IsActiveMember_2":"IsActiveMember_No"},inplace = True)
data_onehot


# In[16]:


data_onehot.rename(columns = {"Geography_France":"France","Geography_Germany":"Germany","Geography_Spain":"Spain",
                            "Gender_Female":"Female","Gender_Male":"Male",
                            "HasCrCard_0":"HasCrCard_No","HasCrCard_1":"HasCrCard_Yes",
                            "IsActiveMember_1":"IsActiveMember_Yes","IsActiveMember_2":"IsActiveMember_No"},inplace = True)
data_onehot


# In[23]:


corr = data_onehot.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='rocket', annot=True, fmt='.1f')
plt.title('Correlation Heatmap with Numerical Values')
plt.show()


# # Split train and test and fix imbalance train set

# In[24]:


X = data_onehot.drop(columns = ["Exited"]).copy()
y = data_onehot["Exited"]
X_train_ori, X_test, y_train_ori, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[25]:


print("X_train shape : ",X_train_ori.shape)
X_train_ori.head()


# In[26]:


print("X_test shape : ",X_test.shape)
X_test.head()


# In[27]:


X_train, y_train = ADASYN(random_state=0).fit_resample(X_train_ori, y_train_ori)


# In[28]:


print("X_train orginal shape : ",X_train_ori.shape)
print("X_train shape after ADASYN : ",X_train.shape)
print("y_train orginal shape : ",y_train_ori.shape)
print("y_train shape after ADASYN : ",y_train.shape)


# In[29]:


d1 = pd.DataFrame(y_train_ori.value_counts())
plt.figure(figsize=(10,6))
pal = sns.color_palette("pastel")
plt.pie(d1['Exited'], labels = ["Not Exited", "Exited"], colors=pal, autopct='%1.1f%%', textprops={'fontsize': 14})

plt.title("Exited counts of imbalance train set", fontsize = 15)
plt.legend(bbox_to_anchor=(1, 1) , prop={'size': 15})
plt.tight_layout()

plt.show()


# In[30]:


d2 = pd.DataFrame(y_train.value_counts())
plt.figure(figsize=(10,6))
pal = sns.color_palette("pastel")
plt.pie(d2['Exited'], labels = ["Not Exited", "Exited"], colors=pal, autopct='%1.1f%%',
            textprops={'fontsize': 14}, startangle=90)

plt.title("Exited counts of balance train set", fontsize = 15)
plt.legend(bbox_to_anchor=(1, 1) , prop={'size': 15})
plt.tight_layout()

plt.show()


# # Standard Scaler

# In[31]:


X_train_scale = X_train.copy()
X_test_scale = X_test.copy()


# In[32]:


con_col = ["CreditScore","Age","Balance","EstimatedSalary"]

scaler = StandardScaler()
scaler.fit(X_train[con_col])
X_train_scale[con_col] = scaler.transform(X_train[con_col])
X_train_scale.head()


# In[33]:


X_test_scale[con_col] = scaler.transform(X_test[con_col])
X_test_scale.head()


# # 1. K-nearest neighbors

# In[80]:


knn_parameters = {
    'n_neighbors': list(range(1,10)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_parameters, cv=10)
knn_cv.fit(X_train_scale,y_train)


# In[31]:


print("Best parameters : ",knn_cv.best_params_)
print("Test accuracy :  {:.3f}".format(knn_cv.best_score_))
print("Train accuracy : {:.3f}".format(knn_cv.score(X_test_scale,y_test)))


# In[36]:


knn_y_pred = knn_cv.predict(X_test_scale)
knn_cm = confusion_matrix(y_test, knn_y_pred)

f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(knn_cm,annot = True, linewidths= 0.5, fmt=".0f", ax=ax, cmap = "viridis")
plt.title("K-nearest neighbors confusion matrix after GridSearchCV")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

print("KNN_Precision_score :  {:.3f}".format(precision_score(y_test,knn_y_pred)))
print("KNN_Recall_score :  {:.3f}".format(recall_score(y_test,knn_y_pred)))
print("KNN_F1_score :  {:.3f}".format(f1_score(y_test,knn_y_pred)))


# # 2. RandomForest

# In[73]:


param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90,100],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : list(range(1,5)),
    'criterion' :['gini', 'entropy']
}
forest_cv = RandomForestClassifier(random_state = 1) 
forest_cv = GridSearchCV(estimator=forest_cv, param_grid=param_grid, cv= 10)
forest_cv.fit(X_train_scale, y_train)


# In[74]:


print("Best parameters : ",forest_cv.best_params_)
print("Test accuracy :  {:.3f}".format(forest_cv.best_score_))
print("Train accuracy : {:.3f}".format(forest_cv.score(X_test_scale,y_test)))


# In[75]:


forest_y_pred = forest_cv.predict(X_test_scale)
forest_y_true = y_test
forest_cm = confusion_matrix(forest_y_true, forest_y_pred)

f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(forest_cm,annot = True, linewidths= 0.5, fmt=".0f", ax=ax, cmap = "viridis")
plt.title("RandomForest confusion matrix after GridSearchCV")
plt.xlabel("Y prediction")
plt.ylabel("Y")
plt.show()

print("RandomForest_Precision_score :  {:.3f}".format(precision_score(y_test,forest_y_pred)))
print("RandomForest_Recall_score :  {:.3f}".format(recall_score(y_test,forest_y_pred)))
print("RandomForest_F1_score :  {:.3f}".format(f1_score(y_test,forest_y_pred)))


# # 3. LogisticRegression

# In[81]:


param_grid = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10]}
lr = LogisticRegression(solver="liblinear", random_state = 1)
lr_cv = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 10)
lr_cv.fit(X_train_scale,y_train)


# In[82]:


print("Best parameters : ",lr_cv.best_params_)
print("Test accuracy :  {:.3f}".format(lr_cv.best_score_))
print("Train accuracy : {:.3f}".format(lr_cv.score(X_test_scale,y_test)))


# In[83]:


print("LogisticRegression_Precision_score :  {:.3f}".format(precision_score(y_test,lr_y_pred)))
print("LogisticRegression_Recall_score :  {:.3f}".format(recall_score(y_test,lr_y_pred)))
print("LogisticRegression_F1_score :  {:.3f}".format(f1_score(y_test,lr_y_pred)))


# In[84]:


lr_y_pred = lr_cv.predict(X_test_scale)
lr_y_true = y_test
lr_cm = confusion_matrix(lr_y_true, lr_y_pred)

f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(lr_cm,annot = True, linewidths= 0.5, fmt=".0f", ax=ax, cmap = "viridis")
plt.title("LogisticRegression confusion matrix after GridSearchCV")
plt.xlabel("Y prediction")
plt.ylabel("Y")
plt.show()

print("LogisticRegression_Precision_score :  {:.3f}".format(precision_score(y_test,lr_y_pred)))
print("LogisticRegression_Recall_score :  {:.3f}".format(recall_score(y_test,lr_y_pred)))
print("LogisticRegression_F1_score :  {:.3f}".format(f1_score(y_test,lr_y_pred)))


# # 4. Decision Tree

# In[70]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [10, 20, 30, 40, 50],
     'max_features': ['sqrt', 'log2'],
     'min_samples_leaf': [1, 2, 3, 4, 5]}

tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train_scale,y_train)


# In[71]:


print("Best parameters : ",tree_cv.best_params_)
print("Test accuracy :  {:.3f}".format(tree_cv.best_score_))
print("Train accuracy : {:.3f}".format(tree_cv.score(X_test_scale,y_test)))


# In[72]:


print("DecisionTree_Precision_score :  {:.3f}".format(precision_score(y_test,tree_y_pred)))
print("DecisionTree_Recall_score :  {:.3f}".format(recall_score(y_test,tree_y_pred)))
print("DecisionTree_F1_score :  {:.3f}".format(f1_score(y_test,tree_y_pred)))


# In[85]:


tree_y_pred = tree_cv.predict(X_test_scale)
tree_y_true = y_test
tree_cm = confusion_matrix(tree_y_true, tree_y_pred)

f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(tree_cm,annot = True, linewidths= 0.5, fmt=".0f", ax=ax, cmap = "viridis")
plt.title("LogisticRegression confusion matrix after GridSearchCV")
plt.xlabel("Y prediction")
plt.ylabel("Y")
plt.show()

print("DecisionTree_Precision_score :  {:.3f}".format(precision_score(y_test,tree_y_pred)))
print("DecisionTree_Recall_score :  {:.3f}".format(recall_score(y_test,tree_y_pred)))
print("DecisionTree_F1_score :  {:.3f}".format(f1_score(y_test,tree_y_pred)))


# # 5. SVC

# In[86]:


param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

svc = SVC()
svc_cv = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
svc_cv.fit(X_train_scale,y_train)


# In[87]:


print("Best parameters : ",svc_cv.best_params_)
print("Test accuracy :  {:.3f}".format(svc_cv.best_score_))
print("Train accuracy : {:.3f}".format(svc_cv.score(X_test_scale,y_test)))


# In[88]:


svc_y_pred = svc_cv.predict(X_test_scale)
tree_cm = confusion_matrix(y_test, svc_y_pred)

f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(tree_cm,annot = True, linewidths= 0.5, fmt=".0f", ax=ax, cmap = "viridis")
plt.title("SVC confusion matrix after GridSearchCV")
plt.xlabel("Y prediction")
plt.ylabel("Y")
plt.show()

print("SVC_Precision_score :  {:.3f}".format(precision_score(y_test,svc_y_pred)))
print("SVC_Recall_score :  {:.3f}".format(recall_score(y_test,svc_y_pred)))
print("SVC_F1_score :  {:.3f}".format(f1_score(y_test,svc_y_pred)))


# # Classification report

# 1. K-nearest neighbors

# In[89]:


print(classification_report(y_test,knn_y_pred))


# 2. RandomForest

# In[90]:


print(classification_report(forest_y_true,forest_y_pred))


# 3. LogisticRegression

# In[91]:


print(classification_report(lr_y_true,lr_y_pred))


# 4. DecistionTree

# In[92]:


print(classification_report(lr_y_true,tree_y_pred))


# 5. SVC

# In[93]:


print(classification_report(y_test,svc_y_pred))


# # Result

# In[94]:


print("Model Recall on test set")
d1 = {'Model': ["K-nearest neighbors","RandomForest","LogisticRegression",'DecisionTree','SVC'], 
      'Test Accuracy': [knn_cv.best_score_,forest_cv.best_score_,lr_cv.best_score_,tree_cv.best_score_,svc_cv.best_score_],
    'Train Accuracy': [knn_cv.score(X_test_scale,y_test),forest_cv.score(X_test_scale,y_test),
                       lr_cv.score(X_test_scale,y_test),tree_cv.score(X_test_scale,y_test),svc_cv.score(X_test_scale,y_test)],
    'F-1 Score': [f1_score(y_test,knn_y_pred),f1_score(y_test,forest_y_pred),f1_score(y_test,lr_y_pred),
                  f1_score(y_test,tree_y_pred),f1_score(y_test,svc_y_pred)]}
    
df1 = pd.DataFrame(data=d1)
pd.set_option("display.max_colwidth", 10000)
df1 = df1.set_index('Model')
#df1.sort_values(by = ["Train Accuracy"],inplace=True,ascending=False)
df1 = df1.sort_values(by='Train Accuracy', ascending=False)
df1


# In[98]:


models = df1.index
test_accuracy = df1['Test Accuracy']
train_accuracy = df1['Train Accuracy']
bar_width = 0.35
index = np.arange(len(models))

# สร้างกราฟแท่งแนวนอนแบบติดกัน
plt.figure(figsize=(12, 8))
plt.barh(index - bar_width/2, test_accuracy, bar_width, label='Test Accuracy', alpha=0.7)
plt.barh(index + bar_width/2, train_accuracy, bar_width, label='Train Accuracy', alpha=0.5)

# เพิ่มค่าจำนวนบนแต่ละแท่ง
for i, v in enumerate(test_accuracy):
    plt.text(v + 0.01, i - bar_width/2, f'{v:.2f}', color='black', va='center', fontsize=12)
for i, v in enumerate(train_accuracy):
    plt.text(v + 0.01, i + bar_width/2, f'{v:.2f}', color='black', va='center', fontsize=12)

# เพิ่มรายละเอียดกราฟ
plt.xlabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.yticks(index, models)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# แสดงกราฟ
plt.tight_layout()
plt.show()


# In[96]:


models = df1.index
test_accuracy = df1['Test Accuracy']
train_accuracy = df1['Train Accuracy']
bar_width = 0.35
index = np.arange(len(models))


plt.figure(figsize=(12, 6))
plt.bar(index - bar_width/2, test_accuracy, bar_width, label='Test Accuracy', alpha=0.7)
plt.bar(index + bar_width/2, train_accuracy, bar_width, label='Train Accuracy', alpha=0.7)


for i, v in enumerate(test_accuracy):
    plt.text(i - bar_width/2, v + 0.01, f'{v:.2f}', color='black', ha='center')
for i, v in enumerate(train_accuracy):
    plt.text(i + bar_width/2, v + 0.01, f'{v:.2f}', color='black', ha='center')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.xticks(index, models)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# __เราจะเห็นได้ว่าโมเดลที่เหมาะสมที่สุดกับข้อมูลก็คือ SVC โดยมีค่า train accuracy เท่ากับ 0.8590 และค่า F1-score เท่ากับ 0.6094 ซึ่งสูงที่สุดจากทั้ง 5 โมเดล รองลงมาก็คือ K-nearest neighbors (k-NN) ซึ่งมีประสิทธิภาพใกล้เคียงจาก SVC  จึงเป็นตัวเลือกที่ดีถัดจาก SVC__

# __We can see that the model that best fits the data is SVC, with a train accuracy value of 0.8590 and an F1-score value of 0.6094, which is the highest of the 5 models, followed by K-nearest neighbors (k-NN), which It has similar performance to SVC, making it a good choice next to SVC.__
