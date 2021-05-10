import ipfn
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import requests
import time
import datetime
import sklearn.linear_model
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
pio.renderers.default='browser'

bpm=['36005','36047','36061','36081','36085','36059','36103','36119','36087','36079','36071','36027',
     '09001','09009','34017','34003','34031','34013','34039','34027','34035','34023','34025','34029',
     '34037','34041','34019','34021']
geoxwalk=pd.read_csv(path+'POP/GEOIDCROSSWALK.csv',dtype=str)
bpmpuma=geoxwalk.loc[np.isin(geoxwalk['StateCounty'],bpm),'PUMA2010'].unique()
bpmct=geoxwalk.loc[np.isin(geoxwalk['StateCounty'],bpm),'CensusTract2010'].unique()




quadstatebkpt=gpd.read_file('C:/Users/mayij/Desktop/DOC/DCP2018/TRAVELSHEDREVAMP/shp/quadstatebkpt.shp')
quadstatebkpt.crs=4326

# Res Tract Centroid
res=[]
for i in ['ct','nj','ny']:
    res+=[pd.read_csv(path+'LEHD/'+i+'_rac_S000_JT01_2018.csv',dtype=float,converters={'h_geocode':str})]
res=pd.concat(res,axis=0,ignore_index=True)
res['RESCOUNTY']=[str(x)[0:5] for x in res['h_geocode']]
res['RESCT']=[str(x)[0:11] for x in res['h_geocode']]
res=res[np.isin(res['RESCOUNTY'],bpm)].reset_index(drop=True)
res=pd.merge(res,quadstatebkpt,how='inner',left_on='h_geocode',right_on='blockid')
res['lattt']=res['C000']*res['lat']
res['longtt']=res['C000']*res['long']
res=res.groupby(['RESCT'],as_index=False).agg({'lattt':'sum','longtt':'sum','C000':'sum'}).reset_index(drop=True)
res['RESLAT']=res['lattt']/res['C000']
res['RESLONG']=res['longtt']/res['C000']
res=res[['RESCT','RESLAT','RESLONG']].reset_index(drop=True)

# Work Tract Centroid
work=[]
for i in ['ct','nj','ny']:
    work+=[pd.read_csv(path+'LEHD/'+i+'_wac_S000_JT01_2018.csv',dtype=float,converters={'w_geocode':str})]
work=pd.concat(work,axis=0,ignore_index=True)
work['WORKCOUNTY']=[str(x)[0:5] for x in work['w_geocode']]
work['WORKCT']=[str(x)[0:11] for x in work['w_geocode']]
work=work[np.isin(work['WORKCOUNTY'],bpm)].reset_index(drop=True)
work=pd.merge(work,quadstatebkpt,how='inner',left_on='w_geocode',right_on='blockid')
work['lattt']=work['C000']*work['lat']
work['longtt']=work['C000']*work['long']
work=work.groupby(['WORKCT'],as_index=False).agg({'lattt':'sum','longtt':'sum','C000':'sum'}).reset_index(drop=True)
work['WORKLAT']=work['lattt']/work['C000']
work['WORKLONG']=work['longtt']/work['C000']
work=work[['WORKCT','WORKLAT','WORKLONG']].reset_index(drop=True)


# Skim
skim=pd.DataFrame(ipfn.ipfn.product(res['RESCT'].unique(),work['WORKCT'].unique()),columns=['RESCT','WORKCT'])
skim=pd.merge(skim,res,how='inner',on='RESCT')
skim=pd.merge(skim,work,how='inner',on='WORKCT')
skim=skim[['RESCT','RESLAT','RESLONG','WORKCT','WORKLAT','WORKLONG']].reset_index(drop=True)
skim['DIST']=np.nan
skim['CAR']=np.nan
skim.to_csv(path+'SKIM/skim.csv',index=False)




skim=pd.read_csv(path+'SKIM/skim.csv',dtype=str)
doserver='http://159.65.64.166:8801/'
cutoffinterval=2 # in minutes
cutoffstart=0
cutoffend=60
cutoffincrement=cutoffstart
cutoff=''
while cutoffincrement<cutoffend:
    cutoff+='&cutoffSec='+str((cutoffincrement+cutoffinterval)*60)
    cutoffincrement+=cutoffinterval


i=20000000
skim.loc[i]
url=doserver+'otp/routers/default/isochrone?batch=true&mode=CAR'
url+='&fromPlace='+str(skim.loc[i,'RESLAT'])+','+str(skim.loc[i,'RESLONG'])
url+=cutoff
headers={'Accept':'application/json'}  
req=requests.get(url=url,headers=headers)
js=req.json()
iso=gpd.GeoDataFrame.from_features(js,crs={4326})
bk['T'+arrt[0:2]+arrt[3:5]]=999
cut=range(cutoffend,cutoffstart,-cutoffinterval)


start=datetime.datetime.now()
for i in skim.index[0:100]:
    url=doserver+'otp/routers/default/plan?fromPlace='
    url+=str(skim.loc[i,'RESLAT'])+','+str(skim.loc[i,'RESLONG'])
    url+='&toPlace='+str(skim.loc[i,'WORKLAT'])+','+str(skim.loc[i,'WORKLONG'])+'&mode=CAR'
    headers={'Accept':'application/json'}
    req=requests.get(url=url,headers=headers)
    js=req.json()
    if list(js.keys())[1]=='error':
        skim.loc[i,'DIST']=np.nan
        skim.loc[i,'CAR']=np.nan
    else:
        skim.loc[i,'DIST']=js['plan']['itineraries'][0]['legs'][0]['distance']
        skim.loc[i,'CAR']=js['plan']['itineraries'][0]['legs'][0]['duration']
    time.sleep(0.1)
print(datetime.datetime.now()-start)






rhtstrip=pd.read_csv(path+'RHTS/LINKED_Public.csv',dtype=str)
rhtstrip['TRIPID']=rhtstrip['SAMPN']+'|'+rhtstrip['PERNO']+'|'+rhtstrip['PLANO']
rhtstrip['TOURID']=rhtstrip['SAMPN']+'|'+rhtstrip['PERNO']+'|'+rhtstrip['TOUR_ID']
rhtstrip['PPID']=rhtstrip['SAMPN']+'|'+rhtstrip['PERNO']
rhtstrip['HHID']=rhtstrip['SAMPN'].copy()
rhtstrip=rhtstrip[rhtstrip['ODTPURP']=='1'].reset_index(drop=True)
rhtstrip['RESCT']=[str(x).zfill(11) for x in rhtstrip['OTRACT']]
rhtstrip['WORKCT']=[str(x).zfill(11) for x in rhtstrip['DTRACT']]
rhtstrip['DIST']=pd.to_numeric(rhtstrip['TRPDIST_HN'])
rhtstrip=rhtstrip[['PPID','RESCT','WORKCT','DIST']].drop_duplicates(keep='first').reset_index(drop=True)
rhtspp=pd.read_csv(path+'RHTS/rhtspp.csv',dtype=str,converters={'PWGTP':float})
rhtstrip=pd.merge(rhtstrip,rhtspp,how='inner',on='PPID')
rhtstrip=rhtstrip[['PPID','PPIND','RESCT','WORKCT']].drop_duplicates(keep='first').reset_index(drop=True)
rhtstrip=rhtstrip[~np.isin(rhtstrip['PPIND'],['U16','NLF','CUP'])].reset_index(drop=True)
wacct=pd.read_csv(path+'LEHD/wacct.csv',dtype=float,converters={'CT':str})
wacct=wacct[['CT','AGR','EXT','UTL','CON','MFG','WHL','RET','TRN','INF','FIN','RER','PRF','MNG','WMS','EDU',
             'MED','ENT','ACC','SRV','ADM']].reset_index(drop=True)
wacct=wacct.melt(id_vars=['CT'],value_vars=['AGR','EXT','UTL','CON','MFG','WHL','RET','TRN','INF','FIN','RER',
                                            'PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM'],var_name='PPIND',value_name='WAC')
wacct=wacct[wacct['WAC']>100].reset_index(drop=True)
wacct.columns=['WORKCT','PPIND','WAC']
rhtstrip=pd.merge(rhtstrip,wacct,how='inner',on=['PPIND'])
rhtstrip['DEC']=np.where(rhtstrip['WORKCT_x']==rhtstrip['WORKCT_y'],1,0)



k=rhtstrip.loc[:,:].reset_index(drop=True)
k.DEC.value_counts()
k=k[['PPID','PPIND','RESCT','WORKCT_y','WAC','DEC']].reset_index(drop=True)
k.columns=['PPID','PPIND','RESCT','WORKCT','WAC','DEC']
s=k[['RESCT','WORKCT']].drop_duplicates(keep='first').reset_index(drop=True)
s=pd.merge(s,res,how='inner',on='RESCT')
s=pd.merge(s,work,how='inner',on='WORKCT')
s=s[['RESCT','RESLAT','RESLONG','WORKCT','WORKLAT','WORKLONG']].reset_index(drop=True)
# s['DIST']=np.nan
# s['CAR']=np.nan
s['DIST']=np.sqrt((s['RESLAT']-s['WORKLAT'])**2+(s['RESLONG']-s['WORKLONG'])**2)
# s.to_csv(path+'SKIM/s.csv',index=False)
s=s[['RESCT','WORKCT','DIST']].reset_index(drop=True)
k=pd.merge(k,s,how='inner',on=['RESCT','WORKCT'])



# MNL
reg=sklearn.linear_model.LogisticRegression().fit(k[['DIST','WAC']],k['DEC'])
sm.MNLogit(k['DEC'],sm.add_constant(k[['DIST','WAC']])).fit().summary()
ypred=pd.DataFrame({'train':k['DEC'],'pred':reg.predict(k[['DIST','WAC']]),
                    'prob':[x[1] for x in reg.predict_proba(k[['DIST','WAC']])],
                    'problog':[x[1] for x in reg.predict_log_proba(k[['DIST','WAC']])]})
print(sklearn.metrics.classification_report(ypred['train'],ypred['pred']))



k['PROB']=[x[1] for x in reg.predict_proba(k[['DIST','WAC']])]
l=k.groupby('PPID').agg({'PROB':'sum'})
k=pd.merge(k,l,how='inner',on='PPID')
k['PROB']=k['PROB_x']/k['PROB_y']
k=k.sort_values(['PROB'],ascending=False).reset_index(drop=True)
k=k.drop_duplicates(['PPID'],keep='first').reset_index(drop=True)
k.groupby('WORKCT').count().sort_values(['PPID'],ascending=False)







s=pd.read_csv(path+'SKIM/s.csv',dtype=str)
doserver='http://159.65.64.166:8801/'
start=datetime.datetime.now()
for i in s.index:
    url=doserver+'otp/routers/default/plan?fromPlace='
    url+=str(s.loc[i,'RESLAT'])+','+str(s.loc[i,'RESLONG'])
    url+='&toPlace='+str(s.loc[i,'WORKLAT'])+','+str(s.loc[i,'WORKLONG'])+'&mode=CAR'
    headers={'Accept':'application/json'}
    req=requests.get(url=url,headers=headers)
    js=req.json()
    if list(js.keys())[1]=='error':
        s.loc[i,'DIST']=np.nan
        s.loc[i,'CAR']=np.nan
    else:
        s.loc[i,'DIST']=js['plan']['itineraries'][0]['legs'][0]['distance']
        s.loc[i,'CAR']=js['plan']['itineraries'][0]['legs'][0]['duration']
    time.sleep(0.1)
print(datetime.datetime.now()-start)
s.to_csv(path+'SKIM/s.csv',index=False)




