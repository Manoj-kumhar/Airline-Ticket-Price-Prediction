#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data=pd.read_excel(r"G:\airline/Data_Train.xlsx")


# In[3]:


train_data.head(7)


# In[4]:


train_data.info()


# In[5]:


train_data.isnull().sum()


# In[6]:


train_data.shape


# In[7]:


train_data[train_data['Total_Stops'].isnull()]


# In[8]:


train_data.dropna(inplace=True)


# In[9]:


train_data.isnull().sum()


# In[10]:


data=train_data.copy()


# In[11]:


data.head(2)


# In[12]:


data.dtypes


# In[13]:


def change_into_datetime(col):
    data[col]=pd.to_datetime(data[col])


# In[14]:


data.columns


# In[15]:


for feature in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(feature)


# In[16]:


data.dtypes


# In[17]:


data['Date_of_Journey'].min()


# In[18]:


data['Date_of_Journey'].max()


# In[19]:


data['journey_day']=data['Date_of_Journey'].dt.day


# In[20]:


data['journey_month']=data['Date_of_Journey'].dt.month


# In[21]:


data['journey_year']=data['Date_of_Journey'].dt.year


# In[22]:


data.head(2)


# In[23]:


data.drop('Date_of_Journey',axis=1,inplace=True)


# In[24]:


data.head(2)


# In[25]:


def extract_hour_min(df,col):
    df[col+'_hour']=df[col].dt.hour
    df[col+'_minute']=df[col].dt.minute
    df.drop(col,axis=1,inplace=True)
    return df.head(2)


# In[26]:



extract_hour_min(data,'Dep_Time')


# In[27]:


extract_hour_min(data,'Arrival_Time')


# In[28]:


def flight_dep_time(x):
    
    if ( x> 4) and (x<=8 ):
        return 'Early mrng'
    
    elif ( x>8 ) and (x<=12 ):
        return 'Morning'
    
    elif ( x>12 ) and (x<=16 ):
        return 'Noon'
    
    elif ( x>16 ) and (x<=20 ):
        return 'Evening'
    
    elif ( x>20 ) and (x<=24 ):
        return 'Night'
    else:
        return 'Late night'


# In[29]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind='bar')


# In[30]:


def preprocess_duration(x):
    if 'h' not in x:
        x='0h '+x
    elif 'm' not in x:
        x=x+' 0m'
    return x
    


# In[31]:


data['Duration']=data['Duration'].apply(preprocess_duration)


# In[32]:


data['Duration']


# In[33]:


data['Duration'][0].split(' ')[0]


# In[34]:


int(data['Duration'][0].split(' ')[0][0:-1])


# In[35]:


int(data['Duration'][0].split(' ')[1][0:-1])


# In[36]:


data['Duration_hours']=data['Duration'].apply(lambda x:int(x.split(' ')[0][0:-1]))


# In[37]:


data['Duration_mins']=data['Duration'].apply(lambda x:int(x.split(' ')[1][0:-1]))


# In[38]:


data.head(3)


# In[39]:


data['Duration_total_mins']=data['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[40]:


data.head(2)


# In[41]:


sns.lmplot(x='Duration_total_mins',y='Price',data=data)


# In[42]:


data['Destination'].unique()


# In[43]:


data['Destination'].value_counts().plot(kind='pie')


# In[44]:


data['Route']


# In[45]:


data[data['Airline']=='Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[46]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Airline',data=data)
plt.xticks(rotation='vertical')


# In[47]:


data.head(4)


# Applying one-hot on data !

# In[48]:


np.round(data['Additional_Info'].value_counts()/len(data)*100,2)


# In[49]:


data.drop(columns=['Additional_Info','Route','Duration_total_mins','journey_year'],axis=1,inplace=True)


# In[50]:


data.columns


# In[51]:


data.head(4)


# In[52]:


cat_col=[col for col in data.columns if data[col].dtype=='object']


# In[53]:


num_col=[col for col in data.columns if data[col].dtype!='object']


# In[54]:


cat_col


# In[55]:


data['Source'].unique()


# In[56]:


for category in data['Source'].unique():
    data['Source_'+category]=data['Source'].apply(lambda x: 1 if x==category else 0)


# In[57]:


data.head(3)


# In[58]:


airlines=data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[59]:


airlines


# In[60]:


dict1={key:index for index,key in enumerate(airlines,0)}


# In[61]:


dict1


# In[62]:


data['Airline']=data['Airline'].map(dict1)


# In[63]:


data['Airline']


# In[64]:


data.head(2)


# In[65]:


data['Destination'].unique()


# In[66]:


data['Destination'].replace('New Delhi','Delhi',inplace=True)


# In[67]:


dest=data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[68]:


dest


# In[69]:


dict2={key:index for index,key in enumerate(dest,0)}


# In[70]:


dict2


# In[71]:


dict2


# In[72]:


data['Destination']=data['Destination'].map(dict2)


# In[73]:


data['Destination']


# In[74]:


data.head(2)


# In[75]:


data['Total_Stops'].unique()


# In[76]:


stops={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[77]:


data['Total_Stops']=data['Total_Stops'].map(stops)


# In[78]:


data['Total_Stops']


# In[79]:


def plot(df,col):
    fig,(ax1,ax2,ax3)=plt.subplots(3,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    sns.distplot(df[col],ax=ax3,kde=False)
    


# In[80]:


plot(data,'Price')


# In[81]:


data['Price']=np.where(data['Price']>=35000,data['Price'].median(),data['Price'])


# In[82]:


plot(data,'Price')


# In[83]:


data.head(2)


# In[84]:


data.drop(columns=['Source','Duration'],axis=1,inplace=True)


# In[85]:


data.head(2)


# In[86]:


data.dtypes


# Performing Feature Selection !
# Unsupported Cell Type. Double-Click to inspect/edit the content.

# In[87]:


from sklearn.feature_selection import mutual_info_regression


# In[88]:


X=data.drop(['Price'],axis=1)


# In[89]:


y=data['Price']


# In[90]:


X.dtypes


# In[91]:


mutual_info_regression(X,y)


# In[92]:


imp=pd.DataFrame(mutual_info_regression(X,y),index=X.columns)
imp.columns=['importance']


# In[93]:


imp.sort_values(by='importance',ascending=False)


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[96]:


from sklearn.ensemble import RandomForestRegressor


# In[97]:


ml_model=RandomForestRegressor()


# In[98]:


model=ml_model.fit(X_train,y_train)


# In[99]:


y_pred=model.predict(X_test)


# In[100]:


y_pred


# In[101]:


y_pred.shape


# In[102]:


len(X_test)


# In[103]:


import pickle


# In[104]:


file=open(r'G:\airline/rf_random.pkl','wb')


# In[105]:


pickle.dump(model,file)


# In[106]:


model=open(r'G:\airline/rf_random.pkl','rb')


# In[107]:


forest=pickle.load(model)


# In[108]:


forest.predict(X_test)


# In[109]:


def mape(y_true,y_pred):
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


# In[110]:


mape(y_test,forest.predict(X_test))


# In[111]:


def predict(ml_model):
    
    model=ml_model.fit(X_train,y_train)
    print('Training_score: {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('Predictions are : {}'.format(y_prediction))
    print('\n')
    
    from sklearn import metrics
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2_score: {}'.format(r2_score))
    print('MSE : ', metrics.mean_squared_error(y_test,y_prediction))
    print('MAE : ', metrics.mean_absolute_error(y_test,y_prediction))
    print('RMSE : ', np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    print('MAPE : ', mape(y_test,y_prediction))
    sns.distplot(y_test-y_prediction)
    


# In[112]:


predict(RandomForestRegressor())


# hypertune ml model
# Hyperparameter Tuning or Hyperparameter Optimization
# 1.Choose following method for hyperparameter tuning
#     a.RandomizedSearchCV --> Fast way to Hypertune model
#     b.GridSearchCV--> Slow way to hypertune my model
# 2.Choose ML algo that u have to hypertune
# 2.Assign hyperparameters in form of dictionary or create hyper-parameter space
# 3.define searching &  apply searching on Training data or  Fit the CV model 
# 4.Check best parameters and best score

# In[113]:


from sklearn.model_selection import RandomizedSearchCV


# In[114]:


reg_rf=RandomForestRegressor()


# In[115]:


np.linspace(start=1000,stop=1200,num=6)


# In[116]:


# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=1000,stop=1200,num=6)]

# Number of features to consider at every split
max_features=["auto", "sqrt"]

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=4)]

# Minimum number of samples required to split a node
min_samples_split=[5,10,15,100]


# In[117]:


# Create the grid or hyper-parameter space
random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split
    
}


# In[118]:


random_grid


# In[119]:


rf_Random=RandomizedSearchCV(reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[120]:


rf_Random.fit(X_train,y_train)


# In[121]:


### to get your best model..
rf_Random.best_params_


# In[122]:


pred2=rf_Random.predict(X_test)


# In[123]:


from sklearn import metrics
metrics.r2_score(y_test,pred2)


# In[ ]:




