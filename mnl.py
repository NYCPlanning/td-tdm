import ipfn
import pandas as pd  
import numpy as np
import sklearn.model_selection
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.neural_network
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
pio.renderers.default = "browser"


# Household
rhtshh=pd.read_csv(path+'RHTS/HH_Public.csv',dtype=str)
rhtshh['HHID']=rhtshh['SAMPN'].copy()
rhtshh['WGTP']=pd.to_numeric(rhtshh['HH_WHT2'])
rhtshh['HHSIZE']=np.where(rhtshh['HHSIZ']=='1','SIZE1',
                 np.where(rhtshh['HHSIZ']=='2','SIZE2',
                 np.where(rhtshh['HHSIZ']=='3','SIZE3','SIZE4')))
rhtshh=rhtshh[rhtshh['INCOM']!='99'].reset_index(drop=True)
rhtshh['HHINC']=np.where(rhtshh['INCOM']=='1','INC201',
                np.where(rhtshh['INCOM']=='2','INC202',
                np.where(rhtshh['INCOM']=='3','INC203',
                np.where(rhtshh['INCOM']=='4','INC204',
                np.where(rhtshh['INCOM']=='5','INC205',
                np.where(rhtshh['INCOM']=='6','INC206',
                np.where(rhtshh['INCOM']=='7','INC207',
                np.where(rhtshh['INCOM']=='8','INC207','OTH'))))))))
rhtshh=rhtshh[rhtshh['RESTY']!='8'].reset_index(drop=True)
rhtshh=rhtshh[rhtshh['RESTY']!='9'].reset_index(drop=True)
rhtshh['HHSTR']=np.where(rhtshh['RESTY']=='1','STR1',
                np.where(rhtshh['RESTY']=='2','STRM',
                np.where(rhtshh['RESTY']=='3','STRO','OTH')))
rhtshh['HHVEH']=np.where(rhtshh['HHVEH']=='0','VEH0',
                np.where(rhtshh['HHVEH']=='1','VEH1',
                np.where(rhtshh['HHVEH']=='2','VEH2',
                np.where(rhtshh['HHVEH']=='3','VEH3','VEH4'))))
rhtshh=rhtshh[['HHID','WGTP','HHSIZE','HHINC','HHSTR','HHVEH']].reset_index(drop=True)

# Person
rhtspp=pd.read_csv(path+'RHTS/PER_Public.csv',dtype=str,encoding='latin-1')
rhtspp['PPID']=rhtspp['SAMPN']+'|'+rhtspp['PERNO']
rhtspp['HHID']=rhtspp['SAMPN'].copy()
rhtspp['PWGTP']=pd.to_numeric(rhtspp['HH_WHT2'])
rhtspp=rhtspp[rhtspp['GENDER']!='RF'].reset_index(drop=True)
rhtspp['PPSEX']=np.where(rhtspp['GENDER']=='Male','MALE','FEMALE')
rhtspp=rhtspp[rhtspp['AGE_R']!='Age not provided'].reset_index(drop=True)
rhtspp['PPAGE']=np.where(rhtspp['AGE_R']=='Younger than 16 years','AGE201',
                np.where(rhtspp['AGE_R']=='16-18 years','AGE202',
                np.where(rhtspp['AGE_R']=='19-24 years','AGE203',
                np.where(rhtspp['AGE_R']=='25-34 years','AGE204',
                np.where(rhtspp['AGE_R']=='35-54 years','AGE205',
                np.where(rhtspp['AGE_R']=='55-64 years','AGE206',
                np.where(rhtspp['AGE_R']=='65 years or older','AGE207','OTHER')))))))
rhtspp=rhtspp[rhtspp['HISP']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['HISP']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['RACE']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['RACE']!='DK'].reset_index(drop=True)
rhtspp['PPRACE']=np.where(rhtspp['HISP']=='Yes','HSP',
                 np.where(rhtspp['RACE']=='White','WHT',
                 np.where(rhtspp['RACE']=='African American, Black','BLK',
                 np.where(rhtspp['RACE']=='American Indian, Alaskan Native','NTV',
                 np.where(rhtspp['RACE']=='Asian','ASN',
                 np.where(rhtspp['RACE']=='Pacific Islander','PCF',
                 np.where(rhtspp['RACE']=='Other (Specify)','OTH',
                 np.where(rhtspp['RACE']=='Multiracial','TWO','OT'))))))))
rhtspp=rhtspp[rhtspp['STUDE']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['STUDE']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['SCHOL']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['SCHOL']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['SCHOL']!='Other (Specify)'].reset_index(drop=True)
rhtspp['PPSCH']=np.where(pd.isna(rhtspp['SCHOL']),'NS',
                np.where(rhtspp['SCHOL']=='Daycare','PR',
                np.where(rhtspp['SCHOL']=='Nursery/Pre-school','PR',
                np.where(rhtspp['SCHOL']=='Kindergarten to Grade 8','G8',
                np.where(rhtspp['SCHOL']=='Grade 9 to 12','HS',
                np.where(rhtspp['SCHOL']=='4-Year College or University','CL',
                np.where(rhtspp['SCHOL']=='2-Year College (Community College)','CL',
                np.where(rhtspp['SCHOL']=='Vocational/Technical School','CL',
                np.where(rhtspp['SCHOL']=='Graduate School/Professional','GS','OT')))))))))
rhtspp=rhtspp[rhtspp['EMPLY']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['EMPLY']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['VOLUN']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['VOLUN']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['WKSTAT']!='RF'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['WKSTAT']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['INDUS']!='DONT KNOW'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['INDUS']!='REFUSED'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['INDUS']!='OTHER (SPECIFY __________)'].reset_index(drop=True)
rhtspp['PPIND']=np.where(pd.isna(rhtspp['EMPLY']),'U16',
                np.where((rhtspp['WORKS']=='Not a Worker')&(rhtspp['WKSTAT']=='Retired'),'NLF',
                np.where((rhtspp['WORKS']=='Not a Worker')&(rhtspp['WKSTAT']=='Homemaker'),'NLF',
                np.where((rhtspp['WORKS']=='Not a Worker')&(rhtspp['WKSTAT']=='Student (Part-time or Full-time)'),'NLF',
                np.where((rhtspp['WORKS']=='Not a Worker')&(rhtspp['WKSTAT']=='Unemployed, Not Seeking Employment'),'NLF',
                np.where((rhtspp['WORKS']=='Not a Worker')&(rhtspp['WKSTAT']=='Unemployed but Looking for Work'),'CUP',
                np.where(rhtspp['OCCUP']=='MILITARY SPECIFIC OCCUPATIONS','MIL',
                np.where(rhtspp['INDUS']=='AGRICULTURE, FORESTRY, FISHING AND HUNTING','AGR',
                np.where(rhtspp['INDUS']=='MINING, QUARRYING, AND OIL AND GAS EXTRACTION','EXT',
                np.where(rhtspp['INDUS']=='CONSTRUCTION','CON',
                np.where(rhtspp['INDUS']=='MANUFACTURING','MFG',
                np.where(rhtspp['INDUS']=='WHOLESALE TRADE','WHL',
                np.where(rhtspp['INDUS']=='RETAIL TRADE','RET',
                np.where(rhtspp['INDUS']=='TRANSPORTATION AND WAREHOUSING','TRN',
                np.where(rhtspp['INDUS']=='UTILITIES','UTL',
                np.where(rhtspp['INDUS']=='INFORMATION','INF',
                np.where(rhtspp['INDUS']=='FINANCE AND INSURANCE','FIN',
                np.where(rhtspp['INDUS']=='REAL ESTATE, RENTAL AND LEASING','RER',
                np.where(rhtspp['INDUS']=='PROFESSIONAL, SCIENTIFIC AND TECHNICAL SERVICES','PRF',
                np.where(rhtspp['INDUS']=='MANAGEMENT OF COMPANIES AND ENTERPRISES','MNG',
                np.where(rhtspp['INDUS']=='ADMINISTRATION AND SUPPORT AND WARE MANAGEMENT AND REMEDIATION SERVICES','WMS',
                np.where(rhtspp['INDUS']=='EDUCATIONAL SERVICES','EDU',
                np.where(rhtspp['INDUS']=='HEALTH CARE AND SOCIAL ASSISTANCE','MED',
                np.where(rhtspp['INDUS']=='ARTS, ENTERTAINMENT, AND RECREATION','ENT',
                np.where(rhtspp['INDUS']=='ACCOMODATION AND FOOD SERVICES','ACC',
                np.where(rhtspp['INDUS']=='OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)','SRV',
                np.where(rhtspp['INDUS']=='PUBLIC ADMINISTRATION','ADM','OTH')))))))))))))))))))))))))))
rhtspp=rhtspp[rhtspp['LIC']!='DK'].reset_index(drop=True)
rhtspp=rhtspp[rhtspp['LIC']!='RF'].reset_index(drop=True)
rhtspp['PPLIC']=np.where(pd.isna(rhtspp['LIC']),'U16',
                np.where(rhtspp['LIC']=='Yes','YES',
                np.where(rhtspp['LIC']=='No','NO','OTH')))
rhtspp['PPTRIPS']=np.where(rhtspp['PTRIPS']=='0','TRIP0',
                  np.where(rhtspp['PTRIPS']=='2','TRIP2',
                  np.where(rhtspp['PTRIPS']=='3','TRIP3',
                  np.where(rhtspp['PTRIPS']=='4','TRIP4','TRIPO'))))
# rhtspp['PPTRIPS']=pd.to_numeric(rhtspp['PTRIPS'])
rhtspp=rhtspp[['PPID','HHID','PWGTP','PPSEX','PPAGE','PPRACE','PPSCH','PPIND','PPLIC','PPTRIPS']].reset_index(drop=True)


rhtstrip=pd.read_csv(path+'RHTS/LINKED_Public.csv',dtype=str)
rhtstrip['TOURID']=rhtstrip['SAMPN']+'|'+rhtstrip['PERNO']+'|'+rhtstrip['TOUR_ID']
rhtstrip['TRIPID']=rhtstrip['SAMPN']+'|'+rhtstrip['PERNO']+'|'+rhtstrip['PLANO']

k=rhtstrip.groupby('TRIPID').agg({'PLANO':'count'})

k=rhtstrip[(rhtstrip['SAMPN']=='4133756')&(rhtstrip['PERNO']=='2')&(rhtstrip['TOUR_ID']=='1')]





df=pd.merge(rhtspp,rhtshh,how='inner',on='HHID')
df['PPIND']=np.where(df['PPIND']=='U16','U16',
            np.where(df['PPIND']=='NLF','NLF',
            np.where(df['PPIND']=='CUP','CUP','EMP')))

xtrain=pd.get_dummies(df[['PPIND']],drop_first=True)
ytrain=df[['PPTRIPS']]
sm.MNLogit(ytrain,sm.add_constant(xtrain)).fit().summary()


df=df.groupby(['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH'],as_index=False).agg({'PPID':'count'}).reset_index(drop=True)
df.PPID.value_counts(dropna=False)










# LIC
# Data Split
df=pd.merge(rhtspp,rhtshh,how='inner',on='HHID')
xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(df[['PPSEX','PPAGE','PPRACE',
                                                                       'PPSCH','PPIND','HHSIZE',
                                                                       'HHINC','HHSTR','HHVEH']],
                                                                   df['PPLIC'],test_size=0.5)
xtrain=pd.get_dummies(xtrain[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)
xtest=pd.get_dummies(xtest[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)

# NN
nn=sklearn.neural_network.MLPClassifier().fit(xtrain,ytrain)
ypred=pd.DataFrame({'train':ytrain,'pred':nn.predict(xtrain)})
print(sklearn.metrics.classification_report(ytrain,ypred['pred']))
ypred=pd.DataFrame({'test':ytest,'pred':nn.predict(xtest)})
print(sklearn.metrics.classification_report(ytest,ypred['pred']))

# MNL
reg=sklearn.linear_model.LogisticRegression().fit(xtrain,ytrain)
sm.MNLogit(ytrain,sm.add_constant(xtrain)).fit().summary()
ypred=pd.DataFrame({'train':ytrain,'pred':reg.predict(xtrain)})
print(sklearn.metrics.classification_report(ytrain,ypred['pred']))
ypred=pd.DataFrame({'test':ytest,'pred':reg.predict(xtest)})
print(sklearn.metrics.classification_report(ytest,ypred['pred']))





# TRIPS
# Data Split
df=pd.merge(rhtspp,rhtshh,how='inner',on='HHID')
xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(df[['PPSEX','PPAGE','PPRACE',
                                                                       'PPSCH','PPIND','HHSIZE',
                                                                       'HHINC','HHSTR','HHVEH']],
                                                                   df['PPTRIPS'],test_size=0.5)
xtrain=pd.get_dummies(xtrain[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)
xtest=pd.get_dummies(xtest[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)

# NN
nn=sklearn.neural_network.MLPClassifier().fit(xtrain,ytrain)
ypred=pd.DataFrame({'train':ytrain,'pred':nn.predict(xtrain)})
print(sklearn.metrics.classification_report(ytrain,ypred['pred']))
ypred=pd.DataFrame({'test':ytest,'pred':nn.predict(xtest)})
print(sklearn.metrics.classification_report(ytest,ypred['pred']))

# MNL
reg=sklearn.linear_model.LogisticRegression().fit(xtrain,ytrain)
sm.MNLogit(ytrain,sm.add_constant(xtrain)).fit().summary()
ypred=pd.DataFrame({'train':ytrain,'pred':reg.predict(xtrain)})
print(sklearn.metrics.classification_report(ytrain,ypred['pred']))
ypred=pd.DataFrame({'test':ytest,'pred':reg.predict(xtest)})
print(sklearn.metrics.classification_report(ytest,ypred['pred']))








# MNL
xtrain=pd.get_dummies(df[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)
ytrain=df[['PPTRIPS']]
reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(xtrain,ytrain)
sm.MNLogit(ytrain,sm.add_constant(xtrain)).fit().summary()
ypred=pd.DataFrame({'train':ytrain,'pred':reg.predict(xtrain)})
print(sklearn.metrics.classification_report(ytrain,ypred['pred']))

df.PPTRIPS.value_counts(dropna=False)



df=pd.merge(rhtspp,rhtshh,how='inner',on='HHID')
df=df[df['PPTRIPS']<=6].reset_index(drop=True)
xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(df[['PPSEX','PPAGE','PPRACE',
                                                                       'PPSCH','PPIND','HHSIZE',
                                                                       'HHINC','HHSTR','HHVEH']],
                                                                   df['PPTRIPS'],test_size=0.5)
xtrain=pd.get_dummies(xtrain[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)
xtest=pd.get_dummies(xtest[['PPSEX','PPAGE','PPRACE','PPSCH','PPIND','HHSIZE','HHINC','HHSTR','HHVEH']],drop_first=True)
reg=sklearn.linear_model.LinearRegression().fit(xtrain,ytrain)
sm.OLS(ytrain,sm.add_constant(xtrain)).fit().summary()
ypred=pd.DataFrame({'train':ytrain,'pred':[round(x) for x in reg.predict(xtrain)]})
print(sklearn.metrics.classification_report(ytrain,ypred['pred']))
ypred=pd.DataFrame({'test':ytest,'pred':[round(x) for x in reg.predict(xtest)]})
print(sklearn.metrics.classification_report(ytest,ypred['pred']))















# Backup

# pumshh=pd.read_csv(path+'PUMS/pumshh.csv',dtype=str,converters={'WGTP':float})
# pumabed=pumshh[np.isin(pumshh['HHBLT'],['B00','B10','B14'])].reset_index(drop=True)
# pumabed=pumabed.groupby(['PUMA','HHBED'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
# pumabed=pd.merge(pumabed,pumabed.groupby(['PUMA'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True),how='inner',on='PUMA')
# pumabed['PCT']=pumabed['WGTP_x']/pumabed['WGTP_y']
# pumabed=pumabed[['PUMA','HHBED','PCT']].reset_index(drop=True)



# tp=pd.DataFrame(columns=['PUMA','CT','UNIT'])
# tp.loc[0]=['3603805','36005020000',100]
# tpbed=pd.DataFrame(ipfn.ipfn.product(tp['CT'],pumshh['HHBED'].unique()),columns=['CT','HHBED'])
# tpbed=pd.merge(tp,tpbed,how='inner',on='CT')
# tpbed=pd.merge(tpbed,pumabed,how='inner',on=['PUMA','HHBED'])
# tpbed['UNIT']=tpbed['UNIT']*tpbed['PCT']
# tpbed=tpbed[['PUMA','CT','HHBED','UNIT']].reset_index(drop=True)


# # MNL
# df=pd.concat([pumshh[['WGTP','HHSIZE']],pd.get_dummies(pumshh['HHBED'],drop_first=True)],axis=1,ignore_index=False)
# df['WGTP']=pd.to_numeric(df['WGTP'])
# trainx=df.drop(['WGTP','HHSIZE'],axis=1)
# trainy=df['HHSIZE']
# trainwt=df['WGTP']
# reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(trainx,trainy,trainwt)
# # sm.MNLogit(trainy,sm.add_constant(trainx)).fit().summary()

# tpbedsize=pd.DataFrame(ipfn.ipfn.product(sorted(pumshh['HHBED'].unique()),sorted(pumshh['HHSIZE'].unique())),columns=['HHBED','HHSIZE'])
# tpbedsize=pd.concat([tpbedsize[['HHSIZE']],pd.get_dummies(tpbedsize['HHBED'],drop_first=True)],axis=1,ignore_index=False)
# tpbedsize['MNL']=np.nan
# modelx=tpbedsize.drop(['HHSIZE','MNL'],axis=1).drop_duplicates(keep='first')
# tpbedsize['MNL']=np.ndarray.flatten(reg.predict_proba(modelx))
# tpbedsize['HHBED']=np.where(tpbedsize['BED1']==1,'BED1',
#                    np.where(tpbedsize['BED2']==1,'BED2',
#                    np.where(tpbedsize['BED3']==1,'BED3',
#                    np.where(tpbedsize['BED4']==1,'BED4',
#                    np.where(tpbedsize['BED5']==1,'BED5','BED0')))))
# tpbedsize=tpbedsize[['HHBED','HHSIZE','MNL']].reset_index(drop=True)

# # Proportional Allocation
# df=pumshh[['HHBED','HHSIZE','WGTP']].reset_index(drop=True)
# df['WGTP']=pd.to_numeric(df['WGTP'])
# df=df.groupby(['HHBED','HHSIZE'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
# df=pd.merge(df,df.groupby(['HHBED'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True),how='inner',on='HHBED')
# df['PCT']=df['WGTP_x']/df['WGTP_y']
# df=df[['HHBED','HHSIZE','PCT']].reset_index(drop=True)


# tpbedsize=pd.merge(tpbed,tpbedsize,how='inner',on='HHBED')
# tpbedsize=pd.merge(tpbedsize,df,how='inner',on=['HHBED','HHSIZE'])



























# df=pd.read_csv(path+'RHTS/HH_Public.csv')
# df=df[['RESTY','HHSIZ','INCOM','HHVEH','HHSTU','HHWRK','HHLIC','HHCHD','HTRIPS_GPS','HH_WHT2']].reset_index(drop=True)
# df['RESTY']=np.where(df['RESTY']==1,'SG',
#             np.where(df['RESTY']==2,'MT','OT'))
# df['INCOM']=np.where(df['INCOM']==1,'1',
#             np.where(df['INCOM']==2,'2',
#             np.where(df['INCOM']==3,'3',
#             np.where(df['INCOM']==4,'4',
#             np.where(df['INCOM']==5,'5',
#             np.where(df['INCOM']==6,'6',
#             np.where(df['INCOM']==7,'7',
#             np.where(df['INCOM']==8,'8','NA'))))))))
# df['HHVEH']=np.where(df['HHVEH']==0,'0',
#             np.where(df['HHVEH']==1,'1',
#             np.where(df['HHVEH']==2,'2','3+')))
# df=pd.concat([df[['HHSIZ','HHVEH','HHSTU','HHWRK','HHLIC','HHCHD','HTRIPS_GPS','HH_WHT2']],pd.get_dummies(df[['RESTY','INCOM']],drop_first=True)],axis=1,ignore_index=False)



# x=df[['RESTY_OT','RESTY_SG','HHSIZ','INCOM_2','INCOM_3','INCOM_4','INCOM_5','INCOM_6','INCOM_7',
#       'INCOM_8','INCOM_NA','HHSTU','HHWRK','HHLIC','HHCHD']]
# y=df['HHVEH']
# wt=df['HH_WHT2']


# reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(x,y,wt)
# cm=sklearn.metrics.confusion_matrix(y,reg.predict(x))
# sns.heatmap(cm,annot=True)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')





# dt=sklearn.tree.DecisionTreeClassifier().fit(x,y,wt)
# cm=sklearn.metrics.confusion_matrix(y,dt.predict(x))
# sns.heatmap(cm,annot=True)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')



# gbdt=sklearn.ensemble.GradientBoostingClassifier(learning_rate=1).fit(x,y,wt)
# cm=sklearn.metrics.confusion_matrix(y,gbdt.predict(x))
# sns.heatmap(cm,annot=True)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')














# xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(df[['SEX','RACE1','RACE2','RACE3','AGE1','AGE2','AGE3','AGE4','EMP']],df['TRIP'],test_size=0.4)

# reg=sklearn.linear_model.LogisticRegression().fit(xtrain,ytrain)

# ypred=pd.DataFrame({'test':ytest,'pred':reg.predict(xtest)})
# sm.MNLogit(y,sm.add_constant(x)).fit().summary()




# # Confusion Matrix
# cm=sklearn.metrics.confusion_matrix(ytest,reg.predict(xtest))
# sns.heatmap(cm,annot=True)
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
