import pandas as pd  
import numpy as np
import sklearn.model_selection
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
pio.renderers.default = "browser"



df=pd.read_csv(path+'RHTS/HH_Public.csv')
df=df[['RESTY','HHSIZ','INCOM','HHVEH','HHSTU','HHWRK','HHLIC','HHCHD','HTRIPS_GPS','HH_WHT2']].reset_index(drop=True)
df['RESTY']=np.where(df['RESTY']==1,'SG',
            np.where(df['RESTY']==2,'MT','OT'))
df['INCOM']=np.where(df['INCOM']==1,'1',
            np.where(df['INCOM']==2,'2',
            np.where(df['INCOM']==3,'3',
            np.where(df['INCOM']==4,'4',
            np.where(df['INCOM']==5,'5',
            np.where(df['INCOM']==6,'6',
            np.where(df['INCOM']==7,'7',
            np.where(df['INCOM']==8,'8','NA'))))))))
df['HHVEH']=np.where(df['HHVEH']==0,'0',
            np.where(df['HHVEH']==1,'1',
            np.where(df['HHVEH']==2,'2','3+')))
df=pd.concat([df[['HHSIZ','HHVEH','HHSTU','HHWRK','HHLIC','HHCHD','HTRIPS_GPS','HH_WHT2']],pd.get_dummies(df[['RESTY','INCOM']],drop_first=True)],axis=1,ignore_index=False)



x=df[['RESTY_OT','RESTY_SG','HHSIZ','INCOM_2','INCOM_3','INCOM_4','INCOM_5','INCOM_6','INCOM_7',
      'INCOM_8','INCOM_NA','HHSTU','HHWRK','HHLIC','HHCHD']]
y=df['HHVEH']
wt=df['HH_WHT2']


reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(x,y,wt)
cm=sklearn.metrics.confusion_matrix(y,reg.predict(x))
sns.heatmap(cm,annot=True)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')





dt=sklearn.tree.DecisionTreeClassifier().fit(x,y,wt)
cm=sklearn.metrics.confusion_matrix(y,dt.predict(x))
sns.heatmap(cm,annot=True)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')



gbdt=sklearn.ensemble.GradientBoostingClassifier(learning_rate=1).fit(x,y,wt)
cm=sklearn.metrics.confusion_matrix(y,gbdt.predict(x))
sns.heatmap(cm,annot=True)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')














xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(df[['SEX','RACE1','RACE2','RACE3','AGE1','AGE2','AGE3','AGE4','EMP']],df['TRIP'],test_size=0.4)

reg=sklearn.linear_model.LogisticRegression().fit(xtrain,ytrain)

ypred=pd.DataFrame({'test':ytest,'pred':reg.predict(xtest)})
sm.MNLogit(y,sm.add_constant(x)).fit().summary()




# Confusion Matrix
cm=sklearn.metrics.confusion_matrix(ytest,reg.predict(xtest))
sns.heatmap(cm,annot=True)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
