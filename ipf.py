import ipfn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
pio.renderers.default='browser'



ctpuma=pd.read_csv(path+'POP/ctpuma.csv',dtype=str)

pums=pd.read_csv(path+'ACS/psam_h36.csv',dtype=str)
pums=pums.loc[pums['WGTP']!='0',['PUMA','WGTP','NP','HINCP']].reset_index(drop=True)
pums=pums[pums['NP']!='0'].reset_index(drop=True)
pums['PUMA']=[x[1:] for x in pums['PUMA']]
pums['WGTP']=pd.to_numeric(pums['WGTP'])
pums['NP']=pd.to_numeric(pums['NP'])
pums['HHSIZE']=np.where(pums['NP']==1,'SIZE1',
               np.where(pums['NP']==2,'SIZE2',
               np.where(pums['NP']==3,'SIZE3','SIZE4')))
pums['HINCP']=pd.to_numeric(pums['HINCP'])
pums['HHINC']=np.where(pums['HINCP']<5000,'INC01',
              np.where(pums['HINCP']<10000,'INC02',
              np.where(pums['HINCP']<15000,'INC03',
              np.where(pums['HINCP']<20000,'INC04',
              np.where(pums['HINCP']<25000,'INC05',
              np.where(pums['HINCP']<35000,'INC06',
              np.where(pums['HINCP']<50000,'INC07',
              np.where(pums['HINCP']<75000,'INC08',
              np.where(pums['HINCP']<100000,'INC09',
              np.where(pums['HINCP']<150000,'INC10','INC11'))))))))))
pums=pums.groupby(['PUMA','HHSIZE','HHINC'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)







i='3805'

for i in puma:
    hhsize=pd.read_csv(path+'POP/hhsize.csv',dtype=float,converters={'CT':str})
    hhsize=pd.merge(hhsize,ctpuma,how='inner',on='CT')
    hhsize=hhsize[hhsize['PUMA']==i].reset_index(drop=True)
    hhsize=hhsize.melt(id_vars=['CT'],value_vars=['SIZE1','SIZE2','SIZE3','SIZE4'],var_name='HHSIZE',value_name='TOTAL')
    hhsize.groupby('HHSIZE').agg({'TOTAL':'sum'})

    hhinc=pd.read_csv(path+'POP/hhinc.csv',dtype=float,converters={'CT':str})
    hhinc=pd.merge(hhinc,ctpuma,how='inner',on='CT')
    hhinc=hhinc[hhinc['PUMA']==i].reset_index(drop=True)
    hhinc=hhinc.melt(id_vars=['CT'],value_vars=['INC01','INC02','INC03','INC04','INC05','INC06','INC07','INC08','INC09','INC10','INC11'],var_name='HHINC',value_name='TOTAL')
    hhinc.groupby('HHINC').agg({'TOTAL':'sum'})




    hhveh=pd.read_csv(path+'POP/hhveh.csv',dtype=float,converters={'CT':str})
    hhveh=pd.merge(hhveh,ctpuma,how='inner',on='CT')
    hhveh=hhveh[hhveh['PUMA']==i].reset_index(drop=True)
    hhveh=hhveh.melt(id_vars=['CT'],value_vars=['VEH0','VEH1','VEH2','VEH3','VEH4'],var_name='HHVEH',value_name='TOTAL')
    
    hhwork=pd.read_csv(path+'POP/hhwork.csv',dtype=float,converters={'CT':str})
    hhwork=pd.merge(hhwork,ctpuma,how='inner',on='CT')
    hhwork=hhwork[hhwork['PUMA']==i].reset_index(drop=True)
    hhwork=hhwork.melt(id_vars=['CT'],value_vars=['WORK0','WORK1','WORK2','WORK3'],var_name='HHWORK',value_name='TOTAL')

    # pumsszic=pd.DataFrame(ipfn.ipfn.product(hhsize['HHSIZE'].unique(),hhinc['HHINC'].unique()))
    # pumsszic.columns=['HHSIZE','HHINC']
    # pumsszic=pd.merge(pumsszic,pums.loc[pums['PUMA']==i,['HHSIZE','HHINC','WGTP']].reset_index(drop=True),how='left',on=['HHSIZE','HHINC'])
    # pumsszic['WGTP']=pumsszic['WGTP'].fillna(0)
    # pumsszic=pumsszic.set_index(['HHSIZE','HHINC'])
    # pumsszic=pumsszic.iloc[:,0]
    
    df=pd.DataFrame(ipfn.ipfn.product(hhstr['CT'].unique(),hhstr['HHSTR'].unique(),
                                      hhsize['HHSIZE'].unique(),hhinc['HHINC'].unique()))
    df.columns=['CT','HHSTR','HHSIZE','HHINC']
    df['TOTAL']=np.random.randint(0,10,len(df))
    # df['TOTAL']=1
    df=df[['CT','HHSTR','HHSIZE','HHINC','TOTAL']].reset_index(drop=True)

    hhstr=hhstr.set_index(['CT','HHSTR'])
    hhstr=hhstr.iloc[:,0]
    
    hhsize=hhsize.set_index(['CT','HHSIZE'])
    hhsize=hhsize.iloc[:,0]
    
    hhinc=hhinc.set_index(['CT','HHINC'])
    hhinc=hhinc.iloc[:,0]
    
    hhveh=hhveh.set_index(['CT','HHVEH'])
    hhveh=hhveh.iloc[:,0]
    
    hhwork=hhwork.set_index(['CT','HHWORK'])
    hhwork=hhwork.iloc[:,0]
    
    aggregates=[hhstr,hhsize,hhinc]
    dimensions=[['CT','HHSTR'],['CT','HHSIZE'],['CT','HHINC']]
    
    df=ipfn.ipfn.ipfn(df,aggregates,dimensions,weight_col='TOTAL',convergence_rate=1,rate_tolerance=1,
                      max_iteration=100000000000).iteration()
    
    hhstr.groupby('HHSTR').sum()
    df.groupby('HHSTR').sum()
    
    
    df=pd.merge(df,ctpuma,how='inner',on='CT')
    df.to_csv(path+'POP/test1.csv',index=False)



    k=pd.merge(hhsize,df.groupby(['CT','HHSIZE'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['CT','HHSIZE'])
    k['ERR2']=np.square(k['TOTAL_y']-k['TOTAL_x'])
    print(np.sqrt(sum(k['ERR2'])/len(k)))
    p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
    p.show()
    
    k=pd.merge(hhinc,df.groupby(['CT','HHINC'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['CT','HHINC'])
    k['ERR2']=np.square(k['TOTAL_y']-k['TOTAL_x'])
    print(np.sqrt(sum(k['ERR2'])/len(k)))
    p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
    p.show()

    k=pd.merge(pumsszic,df.groupby(['HHSIZE','HHINC'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['HHSIZE','HHINC'])
    k['ERR2']=np.square(k['TOTAL']-k['WGTP'])
    print(np.sqrt(sum(k['ERR2'])/len(k)))
    p=px.scatter(k,x='WGTP',y='TOTAL')
    p.show()
    

    









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































