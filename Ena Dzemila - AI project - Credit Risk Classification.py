#Importing necessary librarires
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.tree import plot_tree


# In[2]:


# importing data from  CSV file
dataset=pd.read_csv('german_credit_data1.csv') 

dataset.shape #displays the shape of the dataset
dataset.sample(10) #displays the details from random 10 rows
dataset.rename(columns={'Unnamed: 0': 'ID'},inplace=True) #renames the first row unnamed to ID
dataset.describe(include='all') #provides an information overview for all columns
print(dataset.head()) #displays the initial details of the data set
target_names = dataset['Risk'].unique().tolist()


# In[3]:


dataset.index #displays the range of index
dataset.set_index('ID',inplace=True) #sets the index to ID so we can ignore tht column later 
dataset.sample(5)


# In[4]:


#Target is Risk with values good or bad. We will set 1 to represent good and 0 bad in terms of risk

risk_mappings={'good':1,'bad':0}
dataset['Risk']=dataset['Risk'].map(risk_mappings) #Using a mapping technique to convert category from string to int
print(dataset.head())


# In[5]:


#converting the age attribute from a continuous to a categorical values 
dataset['Age']=dataset['Age'].fillna('adult') #Takes care of missing values by filling in the Nan as adult
cut_points=[19,30,40,50,60,100] #Divides into categories based on age range i.e. binning
label_names=['adult','middle_age','above_middle_age','old_age','senior_citizen'] #provides a label i.e. categorical value to each range
dataset['Age']=pd.cut(dataset['Age'],cut_points,labels=label_names)
print(dataset.head())


# In[6]:


#There are missing values in column Savings Account, thus we can use the ffill method to manage them

dataset['Saving accounts'].fillna(method='ffill',inplace=True) #replacies the Nan by ffill method
dataset['Saving accounts'].unique() #displays the unique values in the savings account
dataset['Saving accounts'].value_counts() #displays the number of unique values in the savings account 


# In[7]:


#Same missing value replacement steps as with Savings Account, but for Checking account
dataset['Checking account'].fillna(method='bfill',inplace=True) 
dataset['Checking account'].unique()
dataset['Checking account'].value_counts()


# In[8]:


#Converthing String to Integer categorical values as main part of the data cleaning stage
#Age conversion by using mapping technique
age_mappings={'senior_citizen':4, 'adult':0, 'above_middle_age':2, 'old_age':3, 'middle_age':1}
dataset['Age']=dataset['Age'].map(age_mappings)


# In[9]:


#Gender to categorical values
dataset['Sex'].unique()
sex_mappings={'male':1,'female':0}
dataset["Sex"]=dataset["Sex"].map(sex_mappings)


# In[10]:


# Housing to categorical values
dataset["Housing"].unique()
Housing_mappings={'own':2,'free':1,'rent':0}
dataset["Housing"]=dataset['Housing'].map(Housing_mappings)


# In[11]:



# savings account to categorical values
dataset=dataset.dropna()
dataset['Saving accounts'].unique()
saving_mappings={'little':0,'moderate':1,'quite rich':2,'rich':3}
dataset['Saving accounts']=dataset['Saving accounts'].map(saving_mappings)


# In[12]:


# checking account to categorical 
dataset['Checking account'].unique()
dataset['Checking account'].describe()
checking_mappings={'little':0,'moderate':1,'rich':2}
dataset['Checking account']=dataset['Checking account'].map(checking_mappings)


# In[13]:



# conversion of credited amount to categorical values with a pd.cut method wherewe can customize range of our own
cut_points=[0,2000,3000,6000,8000,20000]
labels=['too small','small','big','too big','bigger']
dataset['Credit amount']=pd.cut(dataset['Credit amount'],cut_points,labels=labels)
amount_mappings={'big':2, 'small':1, "too big":3, "bigger":4, 'too small':0}
dataset['Credit amount']=dataset['Credit amount'].map(amount_mappings)


# In[14]:



# Duration to categorical values with a pd.cut method wherewe can customize range of our own
cut_points=[0,10,20,40,60,100]
labels=[0,1,2,3,4]
dataset['Duration']=pd.cut(dataset['Duration'],cut_points,labels=labels)


# In[15]:


#Purpose to categorical
amount_mappings={'radio/TV':1, 'education':0, 'furniture/equipment':2, 'car':3, 'business':4,
       'domestic appliances':5, 'repairs':6, 'vacation/others':7}
dataset['Purpose']=dataset['Purpose'].map(amount_mappings)


# In[17]:


print(dataset.sample(30))


# In[18]:


#Training Step
from sklearn.model_selection import train_test_split

testset=dataset['Risk']
dataset=dataset.drop(['Risk'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(dataset,testset,test_size=0.3,random_state=0) #good practice to split it 30% for test and 70% for train


# In[32]:


#Will try several supervised machine learning algorthims (classification) and evaluate their accuracy for our data set
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
accuracyGaussNB = round(accuracy_score(y_pred, y_test) * 100, 2)
print ("The Accuracy of Gaussian Naive Bayes is:", accuracyGaussNB)


# In[33]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_test)
accuracyDecisionTree = round(accuracy_score(y_pred, y_test) * 100, 2)
print ("The Accuracy of the Decision Tree is:", accuracyDecisionTree)

feature_names = dataset.columns[:9]

print(plot_tree(decisiontree, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True))



# In[34]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
accuracyLogisticRegression = round(accuracy_score(y_pred, y_test) * 100, 2)
print("The Accuracy of the Logistic Regression is:", accuracyLogisticRegression)


# In[35]:


#To preview the accuracies of the three chosen methods in one table, we can create a model
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes',
              'Decision Tree'],
    'Score': [accuracyLogisticRegression, accuracyGaussNB, accuracyDecisionTree]})
models.sort_values(by='Score', ascending=False)




