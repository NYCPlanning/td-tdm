import ipfn
import pandas as pd
import numpy as np

path='C:/Users/mayij/Desktop/DOC/GITHUB/IPFGH/'



sex=pd.read_csv(path+'ACSSEX.csv')
race=pd.read_csv(path+'ACSRACE.csv')
age=pd.read_csv(path+'ACSAGE.csv')
pums=pd.read_csv(path+'PUMS.csv')
pums['SEX']=np.where(pums['SEX']==1,'MALE','FEMALE')
pums['RACE']=np.where(np.isin(pums['RAC1P'],[1]),'WHITE',np.where(np.isin(pums['RAC1P'],[2]),'BLACK',
             np.where(np.isin(pums['RAC1P'],[6]),'ASIAN','OTHER')))
pums['AGE']=np.where(pums['AGEP']<=24,'A<=24',np.where(pums['AGEP']<=34,'A25-34',
            np.where(pums['AGEP']<=54,'A35-54',np.where(pums['AGEP']<=64,'A55-64','A>=65'))))



# df=pd.DataFrame(data=np.repeat(sex['CT'],len(pums['SEX'].unique())*len(pums['RACE'].unique())*len(pums['AGE'].unique()))).reset_index(drop=True)
# df['SEX']=list(np.repeat(['FEMALE','MALE'],len(pums['RACE'].unique())*len(pums['AGE'].unique())))*len(sex['CT'].unique())
# df['RACE']=list(np.repeat(['ASIAN','BLACK','OTHER','WHITE'],len(pums['AGE'].unique())))*len(pums['SEX'].unique())*len(sex['CT'].unique())
# df['AGE']=['A25-34','A35-54','A55-64','A<=24','A>=65']*len(pums['SEX'].unique())*len(pums['RACE'].unique())*len(sex['CT'].unique())
# df['total']=1

df=pd.DataFrame(data=np.repeat(sex['CT'],len(pums['SEX'].unique())*len(pums['AGE'].unique()))).reset_index(drop=True)
df['SEX']=list(np.repeat(['FEMALE','MALE'],len(pums['AGE'].unique())))*len(sex['CT'].unique())
df['AGE']=['A25-34','A35-54','A55-64','A<=24','A>=65']*len(pums['SEX'].unique())*len(sex['CT'].unique())
df['total']=1

# pumasexraceage=df.groupby(['SEX','RACE','AGE'],as_index=False)['total'].sum()
# pumasexraceage['total']=list(pums.groupby(['SEX','RACE','AGE'])['PWGTP'].sum())
# pumasexraceage=pumasexraceage.set_index(['SEX','RACE','AGE'])
# pumasexraceage=pumasexraceage.iloc[:,0]

pumasexage=df.groupby(['SEX','AGE'],as_index=False)['total'].sum()
pumasexage['total']=list(pums.groupby(['SEX','AGE'])['PWGTP'].sum())
pumasexage=pumasexage.set_index(['SEX','AGE'])
pumasexage=pumasexage.iloc[:,0]

ctsex=sex.melt(id_vars=['CT'],value_vars=['FEMALE','MALE'],var_name='SEX',value_name='total')
ctsex=ctsex.set_index(['CT','SEX'])
ctsex=ctsex.iloc[:,0]

ctrace=race.melt(id_vars=['CT'],value_vars=['ASIAN','BLACK','OTHER','WHITE'],
                 var_name='RACE',value_name='total')
ctrace=ctrace.set_index(['CT','RACE'])
ctrace=ctrace.iloc[:,0]

ctage=age.melt(id_vars=['CT'],value_vars=['A25-34','A35-54','A55-64','A<=24','A>=65'],
               var_name='AGE',value_name='total')
ctage=ctage.set_index(['CT','AGE'])
ctage=ctage.iloc[:,0]



# aggregates=[pumasexraceage,ctsex,ctrace,ctage]
# dimensions=[['SEX','RACE','AGE'],['CT','SEX'],['CT','RACE'],['CT','AGE']]

aggregates=[pumasexage,ctsex,ctage]
dimensions=[['SEX','AGE'],['CT','SEX'],['CT','AGE']]

IPF=ipfn.ipfn.ipfn(df,aggregates,dimensions,convergence_rate=1,rate_tolerance=1,max_iteration=100000)
df=IPF.iteration()




k=pd.concat([pumasexage.groupby('SEX').sum(),df.groupby('SEX')['total'].sum()],axis=1)
k=pd.concat([pumasexage.groupby('AGE').sum(),df.groupby('AGE')['total'].sum()],axis=1)
k=pd.concat([pumasexage,df.groupby(['SEX','AGE'])['total'].sum()],axis=1)

# k=pd.concat([pumasexraceage.groupby('SEX').sum(),df.groupby('SEX')['total'].sum()],axis=1)
# k=pd.concat([pumasexraceage.groupby('RACE').sum(),df.groupby('RACE')['total'].sum()],axis=1)
# k=pd.concat([pumasexraceage.groupby('AGE').sum(),df.groupby('AGE')['total'].sum()],axis=1)
# k=pd.concat([pumasexraceage,df.groupby(['SEX','RACE','AGE'])['total'].sum()],axis=1)
k=pd.concat([ctsex,df.groupby(['CT','SEX'])['total'].sum()],axis=1)
k=pd.concat([ctrace,df.groupby(['CT','RACE'])['total'].sum()],axis=1)
k=pd.concat([ctage,df.groupby(['CT','AGE'])['total'].sum()],axis=1)































