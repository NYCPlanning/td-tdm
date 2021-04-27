import ipfn
import pandas as pd
import numpy as np
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



# IPF
pumshh=pd.read_csv(path+'PUMS/pumshh.csv',dtype=str,converters={'WGTP':float})
pumsppgq=pd.read_csv(path+'PUMS/pumsppgq.csv',dtype=str,converters={'PWGTP':float})
pumsppgq=pumsppgq.groupby(['HHID','PUMA'],as_index=False).agg({'PWGTP':'sum'}).reset_index(drop=True)
# i='3603903' 142
for i in bpmpuma:
    ct=geoxwalk.loc[geoxwalk['PUMA2010']==i,'CensusTract2010'].unique()
    # ct=ct[ct!='36085008900']
    
    # Households
    tphh=pumshh[pumshh['PUMA']==i].reset_index(drop=True)
    tphh=pd.DataFrame(ipfn.ipfn.product(tphh['HHID'].unique(),ct),columns=['HHID','CT'])
    tphh=pd.merge(tphh,pumshh[['HHID','HHINC']],how='left',on='HHID')
    tphh['TOTAL']=1
    
    hhwt=pumshh[['HHID','WGTP']].reset_index(drop=True)
    hhwt['WGTP']=pd.to_numeric(hhwt['WGTP'])
    hhwt=hhwt.set_index(['HHID'])
    hhwt=hhwt.iloc[:,0]
    
    cthhinc=pd.read_csv(path+'ACS/hhinc.csv',dtype=float,converters={'CT':str})
    cthhinc=cthhinc.loc[np.isin(cthhinc['CT'],ct),['CT','VAC','INC01','INC02','INC03','INC04','INC05',
                                                   'INC06','INC07','INC08','INC09','INC10','INC11']].reset_index(drop=True)
    cthhinc=cthhinc.melt(id_vars=['CT'],value_vars=['VAC','INC01','INC02','INC03','INC04','INC05','INC06',
                                                    'INC07','INC08','INC09','INC10','INC11'],var_name='HHINC',value_name='TOTAL')
    cthhinc=cthhinc.set_index(['CT','HHINC'])
    cthhinc=cthhinc.iloc[:,0]
    
    aggregates=[hhwt,cthhinc]
    dimensions=[['HHID'],['CT','HHINC']]
    tphh=ipfn.ipfn.ipfn(tphh,aggregates,dimensions,weight_col='TOTAL',max_iteration=1000000).iteration()
    # tphh['TOTAL']=[round(x) for x in tphh['TOTAL']]
    tphh.to_csv(path+'POP/tphh/'+i+'.csv',index=False)
 
    # Group Quarter
    tpgq=pumsppgq[pumsppgq['PUMA']==i].reset_index(drop=True)
    tpgq=pd.DataFrame(ipfn.ipfn.product(tpgq['HHID'].unique(),ct),columns=['HHID','CT'])
    tpgq['TOTAL']=1
    
    gqwt=pumsppgq[['HHID','PWGTP']].reset_index(drop=True)
    gqwt['PWGTP']=pd.to_numeric(gqwt['PWGTP'])
    gqwt=gqwt.set_index(['HHID'])
    gqwt=gqwt.iloc[:,0]
    
    gqtt=pd.read_csv(path+'ACS/gqtt.csv',dtype=float,converters={'CT':str})
    gqtt=gqtt.loc[np.isin(gqtt['CT'],ct),['CT','GQTT']].reset_index(drop=True)
    gqtt=gqtt.set_index(['CT'])
    gqtt=gqtt.iloc[:,0]
    
    aggregates=[gqwt,gqtt]
    dimensions=[['HHID'],['CT']]
    tpgq=ipfn.ipfn.ipfn(tpgq,aggregates,dimensions,weight_col='TOTAL',max_iteration=1000000).iteration()
    # tpgq['TOTAL']=[round(x) for x in tpgq['TOTAL']]
    tpgq.to_csv(path+'POP/tpgq/'+i+'.csv',index=False)

dfhh=[]
for i in bpmpuma:
    tphh=pd.read_csv(path+'POP/tphh/'+i+'.csv',dtype=str,converters={'TOTAL':float})
    dfhh+=[tphh]
dfhh=pd.concat(dfhh,axis=0,ignore_index=True)
dfhh=dfhh.loc[dfhh['TOTAL']!=0,['HHID','CT','HHINC','TOTAL']].reset_index(drop=True)
dfhh['HHGQ']='HH'
dfhh=dfhh.drop_duplicates(keep='first').reset_index(drop=True)
dfhh=dfhh[['HHID','CT','HHGQ','HHINC','TOTAL']].reset_index(drop=True)
dfhh.to_csv(path+'POP/dfhh.csv',index=False)

dfhhgq=[]
for i in bpmpuma:
    tpgq=pd.read_csv(path+'POP/tpgq/'+i+'.csv',dtype=str,converters={'TOTAL':float})
    dfhhgq+=[tpgq]  
dfhhgq=pd.concat(dfhhgq,axis=0,ignore_index=True)
dfhhgq=dfhhgq.loc[dfhhgq['TOTAL']!=0,['HHID','CT','TOTAL']].reset_index(drop=True)
dfhhgq['HHGQ']='GQ'
dfhhgq=dfhhgq.drop_duplicates(keep='first').reset_index(drop=True)
dfhhgq=dfhhgq[['HHID','CT','HHGQ','TOTAL']].reset_index(drop=True)
dfhhgq.to_csv(path+'POP/dfhhgq.csv',index=False)

pumspp=pd.read_csv(path+'PUMS/pumspp.csv',dtype=str,converters={'PWGTP':float})
dfpphh=pd.merge(pumspp,dfhh[['HHID','CT','HHGQ','TOTAL']],how='inner',on=['HHID'])
dfpphh=pd.merge(dfpphh,pumshh,how='inner',on=['HHID','PUMA'])
dfpphh=dfpphh[['PPID','HHID','PUMA','CT','HHGQ','HHSIZE','HHTYPE','HHINC','HHTEN','HHSTR','HHBLT',
               'HHBED','HHVEH','PPSEX','PPAGE','PPRACE','PPEDU','PPSCH','PPIND','PPMODE','TOTAL']].reset_index(drop=True)
pumsppgq=pd.read_csv(path+'PUMS/pumsppgq.csv',dtype=str,converters={'PWGTP':float})
dfppgq=pd.merge(pumsppgq,dfhhgq[['HHID','CT','HHGQ','TOTAL']],how='inner',on=['HHID'])
dfppgq=dfppgq[['PPID','HHID','PUMA','CT','HHGQ','PPSEX','PPAGE','PPRACE','PPEDU','PPSCH','PPIND',
               'PPMODE','TOTAL']].reset_index(drop=True)
dfpp=pd.concat([dfpphh,dfppgq],axis=0,ignore_index=True)
dfpp=dfpp.fillna('GQ')
dfpp.to_csv(path+'POP/dfpp.csv',index=False)



# Validation
# Check Household
pumshh=pd.read_csv(path+'PUMS/pumshh.csv',dtype=str,converters={'WGTP':float})
dfhh=pd.read_csv(path+'POP/dfhh.csv',dtype=str,converters={'TOTAL':float})

# Check HHID Weight Sum
k=pd.merge(dfhh.groupby(['HHID']).agg({'TOTAL':'sum'}),pumshh[['HHID','WGTP']],how='inner',on='HHID')
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='PUMS',
                               x=k['TOTAL'],
                               y=k['WGTP'],
                               mode='markers'))
fig.show()

# Check HHTT
k=pd.read_csv(path+'ACS/hhtt.csv',dtype=float,converters={'CT':str})
k.columns=['CT','TOTAL','MOE']
k=pd.merge(dfhh.groupby(['CT'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHOCC
tptest=dfhh.copy()
tptest['HHOCC']=np.where(tptest['HHINC']=='VAC','VAC','OCC')
k=pd.read_csv(path+'ACS/hhocc.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','OCC','OCCM'],var_name='HHOCC',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHOCC'],['VAC','OCC'])],
           k[np.isin(k['HHOCC'],['VACM','OCCM'])],how='inner',on='CT')
k=k[(k['HHOCC_x']+'M')==k['HHOCC_y']].reset_index(drop=True)
k=k[['CT','HHOCC_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHOCC','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHOCC'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHOCC'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHSIZE
tptest=pd.merge(dfhh,pumshh[['HHID','HHSIZE']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhsize.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','SIZE1','SIZE1M','SIZE2','SIZE2M','SIZE3','SIZE3M',
                                    'SIZE4','SIZE4M'],var_name='HHSIZE',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHSIZE'],['VAC','SIZE1','SIZE2','SIZE3','SIZE4'])],
           k[np.isin(k['HHSIZE'],['VACM','SIZE1M','SIZE2M','SIZE3M','SIZE4M'])],how='inner',on='CT')
k=k[(k['HHSIZE_x']+'M')==k['HHSIZE_y']].reset_index(drop=True)
k=k[['CT','HHSIZE_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHSIZE','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHSIZE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHSIZE'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHTYPE
tptest=pd.merge(dfhh,pumshh[['HHID','HHTYPE']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhtype.csv',dtype=float,converters={'CT':str})
k['TYPE1']=k['MC']-k['MCC']+k['CC']-k['CCC']
k['TYPE1M']=np.sqrt(k['MCM']**2+k['MCCM']**2+k['CCM']**2+k['CCCM']**2)
k['TYPE2']=k['MCC']+k['CCC']
k['TYPE2M']=np.sqrt(k['MCCM']**2+k['CCCM']**2)
k['TYPE3']=k['MHA']+k['FHA']
k['TYPE3M']=np.sqrt(k['MHAM']**2+k['FHAM']**2)
k['TYPE4']=k['MHC']+k['FHC']
k['TYPE4M']=np.sqrt(k['MHCM']**2+k['FHCM']**2)
k['TYPE5']=k['MH']-k['MHC']-k['MHA']+k['FH']-k['FHC']-k['FHA']
k['TYPE5M']=np.sqrt(k['MHM']**2+k['MHCM']**2+k['MHAM']**2+k['FHM']**2+k['FHCM']**2+k['FHAM']**2)
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','TYPE1','TYPE1M','TYPE1','TYPE2M','TYPE3','TYPE3M',
                                    'TYPE4','TYPE4M','TYPE5','TYPE5M'],var_name='HHTYPE',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHTYPE'],['VAC','TYPE1','TYPE2','TYPE3','TYPE4','TYPE5'])],
           k[np.isin(k['HHTYPE'],['VACM','TYPE1M','TYPE2M','TYPE3M','TYPE4M','TYPE5M'])],how='inner',on='CT')
k=k[(k['HHTYPE_x']+'M')==k['HHTYPE_y']].reset_index(drop=True)
k=k[['CT','HHTYPE_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHTYPE','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHTYPE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHTYPE'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHINC
k=pd.read_csv(path+'ACS/hhinc.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','INC01','INC01M','INC02','INC02M','INC03','INC03M',
                                    'INC04','INC04M','INC05','INC05M','INC06','INC06M','INC07','INC07M',
                                    'INC08','INC08M','INC09','INC09M','INC10','INC10M',
                                    'INC11','INC11M'],var_name='HHINC',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHINC'],['VAC','INC01','INC02','INC03','INC04','INC05','INC06','INC07','INC08','INC09','INC10','INC11'])],
           k[np.isin(k['HHINC'],['VACM','INC01M','INC02M','INC03M','INC04M','INC05M','INC06M','INC07M','INC08M','INC09M','INC10M','INC11M'])],how='inner',on='CT')
k=k[(k['HHINC_x']+'M')==k['HHINC_y']].reset_index(drop=True)
k=k[['CT','HHINC_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHINC','TOTAL','MOE']
k=pd.merge(dfhh.groupby(['CT','HHINC'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHINC'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHTEN
tptest=pd.merge(dfhh,pumshh[['HHID','HHTEN']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhten.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','OWNER','OWNERM','RENTER','RENTERM'],var_name='HHTEN',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHTEN'],['VAC','OWNER','RENTER'])],
           k[np.isin(k['HHTEN'],['VACM','OWNERM','RENTERM'])],how='inner',on='CT')
k=k[(k['HHTEN_x']+'M')==k['HHTEN_y']].reset_index(drop=True)
k=k[['CT','HHTEN_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHTEN','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHTEN'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHTEN'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHSTR
tptest=pd.merge(dfhh,pumshh[['HHID','HHSTR']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhstr.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','STR1D','STR1DM','STR1A','STR1AM','STR2','STR2M',
                                    'STR34','STR34M','STR59','STR59M','STR10','STR10M',
                                    'STRO','STROM'],var_name='HHSTR',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHSTR'],['VAC','STR1D','STR1A','STR2','STR34','STR59','STR10','STRO'])],
           k[np.isin(k['HHSTR'],['VACM','STR1DM','STR1AM','STR2M','STR34M','STR59M','STR10M','STROM'])],how='inner',on='CT')
k=k[(k['HHSTR_x']+'M')==k['HHSTR_y']].reset_index(drop=True)
k=k[['CT','HHSTR_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHSTR','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHSTR'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHSTR'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHBLT
tptest=pd.merge(dfhh,pumshh[['HHID','HHBLT']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhblt.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['B14','B14M','B10','B10M','B00','B00M','B90','B90M','B80','B80M',
                                    'B70','B70M','B60','B60M','B50','B50M','B40','B40M','B39','B39M'],var_name='HHBLT',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHBLT'],['B14','B10','B00','B90','B80','B70','B60','B50','B40','B39'])],
           k[np.isin(k['HHBLT'],['B14M','B10M','B00M','B90M','B80M','B70M','B60M','B50M','B40M','B39M'])],how='inner',on='CT')
k=k[(k['HHBLT_x']+'M')==k['HHBLT_y']].reset_index(drop=True)
k=k[['CT','HHBLT_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHBLT','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHBLT'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHBLT'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHBED
tptest=pd.merge(dfhh,pumshh[['HHID','HHBED']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhbed.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['BED0','BED0M','BED1','BED1M','BED2','BED2M','BED3','BED3M',
                                    'BED4','BED4M','BED5','BED5M'],var_name='HHBED',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHBED'],['BED0','BED1','BED2','BED3','BED4','BED5'])],
           k[np.isin(k['HHBED'],['BED0M','BED1M','BED2M','BED3M','BED4M','BED5M'])],how='inner',on='CT')
k=k[(k['HHBED_x']+'M')==k['HHBED_y']].reset_index(drop=True)
k=k[['CT','HHBED_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHBED','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHBED'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHBED'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check HHVEH
tptest=pd.merge(dfhh,pumshh[['HHID','HHVEH']],how='inner',on='HHID')
k=pd.read_csv(path+'ACS/hhveh.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['VAC','VACM','VEH0','VEH0M','VEH1','VEH1M','VEH2','VEH2M',
                                    'VEH3','VEH3M','VEH4','VEH4M'],var_name='HHVEH',value_name='TOTAL')
k=pd.merge(k[np.isin(k['HHVEH'],['VAC','VEH0','VEH1','VEH2','VEH3','VEH4'])],
           k[np.isin(k['HHVEH'],['VACM','VEH0M','VEH1M','VEH2M','VEH3M','VEH4M'])],how='inner',on='CT')
k=k[(k['HHVEH_x']+'M')==k['HHVEH_y']].reset_index(drop=True)
k=k[['CT','HHVEH_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','HHVEH','TOTAL','MOE']
k=pd.merge(tptest.groupby(['CT','HHVEH'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','HHVEH'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))



# Check Group Quarter
pumsppgq=pd.read_csv(path+'PUMS/pumsppgq.csv',dtype=str,converters={'PWGTP':float})
pumsppgq=pumsppgq.groupby(['HHID'],as_index=False).agg({'PWGTP':'sum'}).reset_index(drop=True)
dfhhgq=pd.read_csv(path+'POP/dfhhgq.csv',dtype=str,converters={'TOTAL':float})

# Check GQ Weight Sum
k=pd.merge(dfhhgq.groupby(['HHID']).agg({'TOTAL':'sum'}),pumsppgq[['HHID','PWGTP']],how='inner',on='HHID')
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='PUMS',
                               x=k['TOTAL'],
                               y=k['PWGTP'],
                               mode='markers'))
fig.show()

# Check GQTT
k=pd.read_csv(path+'ACS/gqtt.csv',dtype=float,converters={'CT':str})
k.columns=['CT','TOTAL','MOE']
k=pd.merge(dfhhgq.groupby(['CT'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))



# Check Person
pumspp=pd.read_csv(path+'PUMS/pumspp.csv',dtype=str,converters={'PWGTP':float})
pumsppgq=pd.read_csv(path+'PUMS/pumsppgq.csv',dtype=str,converters={'PWGTP':float})
dfpp=pd.read_csv(path+'POP/dfpp.csv',dtype=str,converters={'PWGTP':float,'TOTAL':float})

# Check PPID Weight Sum
k=pd.merge(dfpp.groupby(['PPID']).agg({'TOTAL':'sum'}),pd.concat([pumspp[['PPID','PWGTP']],pumsppgq[['PPID','PWGTP']]]),how='inner',on='PPID')
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='PUMS',
                               x=k['TOTAL'],
                               y=k['PWGTP'],
                               mode='markers'))
fig.show()

# Check PPTT
k=pd.read_csv(path+'ACS/pptt.csv',dtype=float,converters={'CT':str})
k.columns=['CT','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPSEX
k=pd.read_csv(path+'ACS/ppsex.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['MALE','MALEM','FEMALE','FEMALEM'],var_name='PPSEX',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPSEX'],['MALE','FEMALE'])],
           k[np.isin(k['PPSEX'],['MALEM','FEMALEM'])],how='inner',on='CT')
k=k[(k['PPSEX_x']+'M')==k['PPSEX_y']].reset_index(drop=True)
k=k[['CT','PPSEX_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPSEX','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPSEX'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPSEX'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPAGE
k=pd.read_csv(path+'ACS/ppage.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['AGE01','AGE01M','AGE02','AGE02M','AGE03','AGE03M','AGE04','AGE04M',
                                    'AGE05','AGE05M','AGE06','AGE06M','AGE07','AGE07M','AGE08','AGE08M',
                                    'AGE09','AGE09M','AGE10','AGE10M','AGE11','AGE11M','AGE12','AGE12M',
                                    'AGE13','AGE13M','AGE14','AGE14M','AGE15','AGE15M','AGE16','AGE16M',
                                    'AGE17','AGE17M','AGE18','AGE18M'],var_name='PPAGE',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPAGE'],['AGE01','AGE02','AGE03','AGE04','AGE05','AGE06','AGE07','AGE08',
                                 'AGE09','AGE10','AGE11','AGE12','AGE13','AGE14','AGE15','AGE16',
                                 'AGE17','AGE18'])],
           k[np.isin(k['PPAGE'],['AGE01M''AGE02M','AGE03M','AGE04M','AGE05M','AGE06M','AGE07M',
                                 'AGE08M','AGE09M','AGE10M','AGE11M','AGE12M','AGE13M','AGE14M',
                                 'AGE15M','AGE16M','AGE17M','AGE18M'])],how='inner',on='CT')
k=k[(k['PPAGE_x']+'M')==k['PPAGE_y']].reset_index(drop=True)
k=k[['CT','PPAGE_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPAGE','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPAGE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPAGE'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPRACE
k=pd.read_csv(path+'ACS/pprace.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['HSP','HSPM','WHT','WHTM','BLK','BLKM','NTV','NTVM','ASN','ASNM',
                                    'PCF','PCFM','OTH','OTHM','TWO','TWOM'],var_name='PPRACE',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPRACE'],['HSP','WHT','BLK','NTV','ASN','PCF','OTH','TWO'])],
           k[np.isin(k['PPRACE'],['HSPM','WHTM','BLKM','NTVM','ASNM','PCFM','OTHM','TWOM'])],how='inner',on='CT')
k=k[(k['PPRACE_x']+'M')==k['PPRACE_y']].reset_index(drop=True)
k=k[['CT','PPRACE_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPRACE','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPRACE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPRACE'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPEDU
k=pd.read_csv(path+'ACS/ppedu.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['U18','U18M','U24LH','U24LHM','U24HS','U24HSM','U24AD','U24ADM',
                                    'U24BD','U24BDM','O25G9','O25G9M','O25LH','O25LHM','O25HS','O25HSM',
                                    'O25SC','O25SCM','O25AD','O25ADM','O25BD','O25BDM','O25GD','O25GDM'],var_name='PPEDU',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPEDU'],['U18','U24LH','U24HS','U24AD','U24BD','O25G9','O25LH','O25HS','O25SC',
                                 'O25AD','O25BD','O25GD'])],
           k[np.isin(k['PPEDU'],['U18M','U24LHM','U24HSM','U24ADM','U24BDM','O25G9M','O25LHM','O25HSM',
                                 'O25SCM','O25ADM','O25BDM','O25GDM'])],how='inner',on='CT')
k=k[(k['PPEDU_x']+'M')==k['PPEDU_y']].reset_index(drop=True)
k=k[['CT','PPEDU_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPEDU','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPEDU'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPEDU'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPSCH
k=pd.read_csv(path+'ACS/ppsch.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['NS','NSM','PR','PRM','KG','KGM','G14','G14M','G58','G58M',
                                    'HS','HSM','CL','CLM','GS','GSM'],var_name='PPSCH',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPSCH'],['NS','PR','KG','G14','G58','HS','CL','GS'])],
           k[np.isin(k['PPSCH'],['NSM','PRM','KGM','G14M','G58M','HSM','CLM','GSM'])],how='inner',on='CT')
k=k[(k['PPSCH_x']+'M')==k['PPSCH_y']].reset_index(drop=True)
k=k[['CT','PPSCH_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPSCH','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPSCH'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPSCH'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPIND
k=pd.read_csv(path+'ACS/ppind.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['U16','U16M','AGR','AGRM','EXT','EXTM','CON','CONM','MFG','MFGM',
                                    'WHL','WHLM','RET','RETM','TRN','TRNM','UTL','UTLM','INF','INFM',
                                    'FIN','FINM','RER','RERM','PRF','PRFM','MNG','MNGM','WMS','WMSM',
                                    'EDU','EDUM','MED','MEDM','ENT','ENTM','ACC','ACCM','SRV','SRVM',
                                    'ADM','ADMM','CUP','CUPM','MIL','MILM','NLF','NLFM'],var_name='PPIND',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPIND'],['U16','AGR','EXT','CON','MFG','WHL','RET','TRN','UTL','INF','FIN',
                                 'RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM','CUP',
                                 'MIL','NLF'])],
           k[np.isin(k['PPIND'],['U16M','AGRM','EXTM','CONM','MFGM','WHLM','RETM','TRNM','UTLM','INFM',
                                 'FINM','RERM','PRFM','MNGM','WMSM','EDUM','MEDM','ENTM','ACCM','SRVM',
                                 'ADMM','CUPM','MILM','NLFM'])],how='inner',on='CT')
k=k[(k['PPIND_x']+'M')==k['PPIND_y']].reset_index(drop=True)
k=k[['CT','PPIND_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPIND','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPIND'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPIND'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))

# Check PPMODE
k=pd.read_csv(path+'ACS/ppmode.csv',dtype=float,converters={'CT':str})
k=k.melt(id_vars=['CT'],value_vars=['NW','NWM','DA','DAM','CP2','CP2M','CP3','CP3M','CP4','CP4M',
                                    'CP56','CP56M','CP7','CP7M','BS','BSM','SW','SWM','CR','CRM',
                                    'LR','LRM','FB','FBM','TC','TCM','MC','MCM','BC','BCM','WK','WKM',
                                    'OT','OTM','HM','HMM'],var_name='PPMODE',value_name='TOTAL')
k=pd.merge(k[np.isin(k['PPMODE'],['NW','DA','CP2','CP3','CP4','CP56','CP7','BS','SW','CR','LR','FB',
                                  'TC','MC','BC','WK','OT','HM'])],
           k[np.isin(k['PPMODE'],['NWM','DAM','CP2M','CP3M','CP4M','CP56M','CP7M','BSM','SWM','CRM',
                                  'LRM','FBM','TCM','MCM','BCM','WKM','OTM','HMM'])],how='inner',on='CT')
k=k[(k['PPMODE_x']+'M')==k['PPMODE_y']].reset_index(drop=True)
k=k[['CT','PPMODE_x','TOTAL_x','TOTAL_y']].reset_index(drop=True)
k.columns=['CT','PPMODE','TOTAL','MOE']
k=pd.merge(dfpp.groupby(['CT','PPMODE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPMODE'])
k=k.sort_values('TOTAL_y').reset_index(drop=True)
p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
p.show()
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='ACS',
                               x=list(k.index)+list(k.index[::-1]),
                               y=list(k['TOTAL_y']+k['MOE'])+list((k['TOTAL_y']-k['MOE'])[::-1]),
                               fill='toself',
                               fillcolor='rgba(0,0,255,0.5)',
                               line=dict(color='rgba(255,255,255,0)')))
fig=fig.add_trace(go.Scattergl(name='MODEL',
                               x=k.index,
                               y=k['TOTAL_x'],
                               mode='lines'))
fig.show()
print(len(k[(k['TOTAL_x']<=k['TOTAL_y']+k['MOE'])&(k['TOTAL_x']>=k['TOTAL_y']-k['MOE'])])/len(k))
print(np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k)))































# Backup

# pums=pd.read_csv(path+'POP/psam_h36.csv',dtype=str)
# pums['PUMA']=pums['ST']+pums['PUMA']
# pums=pums.loc[pums['WGTP']!='0',['PUMA','WGTP','NP','HINCP','BLD']].reset_index(drop=True)
# pums=pd.merge(pums,geoxwalk[['PUMA2010','StateCounty']].drop_duplicates(keep='first'),how='left',left_on='PUMA',right_on='PUMA2010')
# pums=pums[np.isin(pums['StateCounty'],bpm)].reset_index(drop=True)
# pums=pums[pums['NP']!='0'].reset_index(drop=True)
# pums['WGTP']=pd.to_numeric(pums['WGTP'])
# pums['NP']=pd.to_numeric(pums['NP'])
# pums['HHSIZE']=np.where(pums['NP']==1,'SIZE1',
#                np.where(pums['NP']==2,'SIZE2',
#                np.where(pums['NP']==3,'SIZE3','SIZE4')))
# pums['HINCP']=pd.to_numeric(pums['HINCP'])
# pums['HHINC']=np.where(pums['HINCP']<5000,'INC01',
#               np.where(pums['HINCP']<10000,'INC02',
#               np.where(pums['HINCP']<15000,'INC03',
#               np.where(pums['HINCP']<20000,'INC04',
#               np.where(pums['HINCP']<25000,'INC05',
#               np.where(pums['HINCP']<35000,'INC06',
#               np.where(pums['HINCP']<50000,'INC07',
#               np.where(pums['HINCP']<75000,'INC08',
#               np.where(pums['HINCP']<100000,'INC09',
#               np.where(pums['HINCP']<150000,'INC10','INC11'))))))))))
# # pums['HHINC']=np.where(pums['HINCP']<25000,'INC201',
# #               np.where(pums['HINCP']<50000,'INC202',
# #               np.where(pums['HINCP']<75000,'INC203',
# #               np.where(pums['HINCP']<100000,'INC204',
# #               np.where(pums['HINCP']<150000,'INC205','INC206')))))
# # pums['HHINC']=np.where(pums['HINCP']<35000,'INC201',
# #               np.where(pums['HINCP']<75000,'INC202',
# #               np.where(pums['HINCP']<150000,'INC203','INC204')))
# pums['HHSTR']=np.where(np.isin(pums['BLD'],['02','03']),'STR1',
#               np.where(np.isin(pums['BLD'],['04','05','06','07','08','09']),'STR2','STR3'))
# pums=pums[['PUMA','StateCounty','HHSIZE','HHINC','HHSTR','WGTP']].reset_index(drop=True)
# # pumstp=pums.groupby(['PUMA','HHSIZE','HHINC'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
# # pums=pd.merge(pums,pumstp,how='inner',on=['PUMA','HHSIZE','HHINC'])
# # pums['WGTP']=pums['WGTP_x'].copy()
# # pums['HHSTRPCT']=pums['WGTP_x']/pums['WGTP_y']
# # pums=pums[['PUMA','HHSIZE','HHINC','HHSTR','WGTP','HHSTRPCT']].reset_index(drop=True)



# predstr=pd.concat([pums[['HHSTR','WGTP']],pd.get_dummies(pums[['PUMA','HHSIZE','HHINC']],drop_first=True)],axis=1,ignore_index=False)
# predstrx=predstr.drop(['HHSTR','WGTP'],axis=1)
# predstry=predstr['HHSTR']
# predstrwt=predstr['WGTP']
# predstrreg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(predstrx,predstry,predstrwt)
# predstrreg.score(predstrx,predstry,predstrwt)
# # ypred=pd.DataFrame({'true':y,'pred':reg.predict(x)})
# # sm.MNLogit(predstry,sm.add_constant(predstrx)).fit().summary()





# i='3603805'

# for i in puma:
#     hhsize=pd.read_csv(path+'POP/hhsize.csv',dtype=float,converters={'CT':str})
#     hhsize=pd.merge(hhsize,geoxwalk[['CensusTract2010','PUMA2010']].drop_duplicates(keep='first'),how='left',left_on='CT',right_on='CensusTract2010')
#     hhsize=hhsize[hhsize['PUMA2010']==i].reset_index(drop=True)
#     hhsize=hhsize.melt(id_vars=['CT'],value_vars=['SIZE1','SIZE2','SIZE3','SIZE4'],var_name='HHSIZE',value_name='TOTAL')
#     hhsize.groupby('HHSIZE').agg({'TOTAL':'sum'})

#     hhinc=pd.read_csv(path+'POP/hhinc.csv',dtype=float,converters={'CT':str})
#     # hhinc['INC201']=hhinc['INC01']+hhinc['INC02']+hhinc['INC03']+hhinc['INC04']+hhinc['INC05'] #<25k
#     # hhinc['INC202']=hhinc['INC06']+hhinc['INC07'] #25k~50k
#     # hhinc['INC203']=hhinc['INC08'].copy() #50k~75k
#     # hhinc['INC204']=hhinc['INC09'].copy() #75k~100k
#     # hhinc['INC205']=hhinc['INC10'].copy() #100k~150k
#     # hhinc['INC206']=hhinc['INC11'].copy() #>150k
#     # hhinc['INC201']=hhinc['INC01']+hhinc['INC02']+hhinc['INC03']+hhinc['INC04']+hhinc['INC05']+hhinc['INC06'] #<35k
#     # hhinc['INC202']=hhinc['INC07']+hhinc['INC08'] #35k~75k
#     # hhinc['INC203']=hhinc['INC09']+hhinc['INC10'] #75k~150k
#     # hhinc['INC204']=hhinc['INC11'].copy() #>150k
#     hhinc=pd.merge(hhinc,geoxwalk[['CensusTract2010','PUMA2010']].drop_duplicates(keep='first'),how='left',left_on='CT',right_on='CensusTract2010')
#     hhinc=hhinc[hhinc['PUMA2010']==i].reset_index(drop=True)
#     hhinc=hhinc.melt(id_vars=['CT'],value_vars=['INC01','INC02','INC03','INC04','INC05','INC06','INC07','INC08','INC09','INC10','INC11'],var_name='HHINC',value_name='TOTAL')
#     # hhinc=hhinc.melt(id_vars=['CT'],value_vars=['INC201','INC202','INC203','INC204','INC205','INC206'],var_name='HHINC',value_name='TOTAL')
#     # hhinc=hhinc.melt(id_vars=['CT'],value_vars=['INC201','INC202','INC203','INC204'],var_name='HHINC',value_name='TOTAL')
#     hhinc.groupby('HHINC').agg({'TOTAL':'sum'})

#     pumsszinc=pd.DataFrame(ipfn.ipfn.product(hhsize['HHSIZE'].unique(),hhinc['HHINC'].unique()))
#     pumsszinc.columns=['HHSIZE','HHINC']
#     pumsszinctp=pums[pums['PUMA']==i].reset_index(drop=True)
#     pumsszinctp=pumsszinctp.groupby(['HHSIZE','HHINC'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
#     pumsszinc=pd.merge(pumsszinc,pumsszinctp,how='left',on=['HHSIZE','HHINC'])
#     pumsszinc['TOTAL']=pumsszinc['WGTP'].fillna(0)
#     pumsszinc=pumsszinc[['HHSIZE','HHINC','TOTAL']].reset_index(drop=True)

#     # pumsszstr=pd.DataFrame(ipfn.ipfn.product(hhsize['HHSIZE'].unique(),hhstr['HHSTR'].unique()))
#     # pumsszstr.columns=['HHSIZE','HHSTR']
#     # pumsszstrtp=pums[pums['PUMA']==i].reset_index(drop=True)
#     # pumsszstrtp=pumsszstrtp.groupby(['HHSIZE','HHSTR'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
#     # pumsszstr=pd.merge(pumsszstr,pumsszstrtp,how='left',on=['HHSIZE','HHSTR'])
#     # pumsszstr['TOTAL']=pumsszstr['WGTP'].fillna(0)
#     # pumsszstr=pumsszstr[['HHSIZE','HHSTR','TOTAL']].reset_index(drop=True)
    
#     # pumsincstr=pd.DataFrame(ipfn.ipfn.product(hhinc['HHINC'].unique(),hhstr['HHSTR'].unique()))
#     # pumsincstr.columns=['HHINC','HHSTR']
#     # pumsincstrtp=pums[pums['PUMA']==i].reset_index(drop=True)
#     # pumsincstrtp=pumsincstrtp.groupby(['HHINC','HHSTR'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
#     # pumsincstr=pd.merge(pumsincstr,pumsincstrtp,how='left',on=['HHINC','HHSTR'])
#     # pumsincstr['TOTAL']=pumsincstr['WGTP'].fillna(0)
#     # pumsincstr=pumsincstr[['HHINC','HHSTR','TOTAL']].reset_index(drop=True) 
    
#     tp=pd.DataFrame(ipfn.ipfn.product(hhsize['CT'].unique(),hhsize['HHSIZE'].unique(),hhinc['HHINC'].unique()))
#     tp.columns=['CT','HHSIZE','HHINC']
#     tp['TOTAL']=np.random.randint(1,10,len(tp))
#     # tp['TOTAL']=1
#     tp=tp[['CT','HHSIZE','HHINC','TOTAL']].reset_index(drop=True)
    
#     hhsize=hhsize.set_index(['CT','HHSIZE'])
#     hhsize=hhsize.iloc[:,0]
    
#     hhinc=hhinc.set_index(['CT','HHINC'])
#     hhinc=hhinc.iloc[:,0]
    
#     pumsszinc=pumsszinc.set_index(['HHSIZE','HHINC'])
#     pumsszinc=pumsszinc.iloc[:,0]

#     aggregates=[hhsize,hhinc,pumsszinc]
#     dimensions=[['CT','HHSIZE'],['CT','HHINC'],['HHSIZE','HHINC']]
    
#     tp=ipfn.ipfn.ipfn(tp,aggregates,dimensions,weight_col='TOTAL',max_iteration=1000000000).iteration()
    
#     k=pd.merge(hhsize,tp.groupby(['CT','HHSIZE'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['CT','HHSIZE'])
#     p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
#     p.show()
    
#     k=pd.merge(hhinc,tp.groupby(['CT','HHINC'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['CT','HHINC'])
#     p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
#     p.show()

#     k=pd.merge(pumsszinc,tp.groupby(['HHSIZE','HHINC'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['HHSIZE','HHINC'])
#     p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
#     p.show()
    
#     tp=pd.concat([tp[['CT','TOTAL']],pd.get_dummies(tp[['HHSIZE','HHINC']],drop_first=True)],axis=1,ignore_index=False)
#     for j in predstrx.columns[:94]:
#         tp[j]=0
#     tp['PUMA_'+i]=1
#     tp['PREDHHSTR']=predstrreg.predict(tp.drop(['CT','TOTAL'],axis=1))
#     tp['HHSIZE']=np.where((tp['HHSIZE_SIZE2']==0)&(tp['HHSIZE_SIZE3']==0)&(tp['HHSIZE_SIZE4']==0),'SIZE1',
#                  np.where((tp['HHSIZE_SIZE2']==1)&(tp['HHSIZE_SIZE3']==0)&(tp['HHSIZE_SIZE4']==0),'SIZE2',
#                  np.where((tp['HHSIZE_SIZE2']==0)&(tp['HHSIZE_SIZE3']==1)&(tp['HHSIZE_SIZE4']==0),'SIZE3',
#                  np.where((tp['HHSIZE_SIZE2']==0)&(tp['HHSIZE_SIZE3']==0)&(tp['HHSIZE_SIZE4']==1),'SIZE4','OTHER'))))
#     tp['HHINC']=np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC01',
#                 np.where((tp['HHINC_INC02']==1)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC02',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==1)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC03',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==1)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC04',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==1)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC05',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==1)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC06',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==1)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC07',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==1)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC08',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==1)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==0),'INC09',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==1)&(tp['HHINC_INC11']==0),'INC10',
#                 np.where((tp['HHINC_INC02']==0)&(tp['HHINC_INC03']==0)&(tp['HHINC_INC04']==0)&(tp['HHINC_INC05']==0)&(tp['HHINC_INC06']==0)&(tp['HHINC_INC07']==0)&(tp['HHINC_INC08']==0)&(tp['HHINC_INC09']==0)&(tp['HHINC_INC10']==0)&(tp['HHINC_INC11']==1),'INC11','OTHER')))))))))))
#     tp=tp[['CT','TOTAL','HHSIZE','HHINC','PREDHHSTR']].reset_index(drop=True)
#     tp.groupby('PREDHHSTR').agg({'TOTAL':'sum'})




#     hhstr=pd.read_csv(path+'POP/hhstr.csv',dtype=float,converters={'CT':str})
#     hhstr=pd.merge(hhstr,geoxwalk[['CensusTract2010','PUMA2010']].drop_duplicates(keep='first'),how='left',left_on='CT',right_on='CensusTract2010')
#     hhstr=hhstr[hhstr['PUMA2010']==i].reset_index(drop=True)

#     k=tp.groupby(['CT','PREDHHSTR'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True)
#     k=k.pivot(index='CT',columns='PREDHHSTR',values='TOTAL').reset_index(drop=False)
#     k=pd.merge(hhstr,k,how='inner',on=['CT'])
#     k['STR1']=np.where((k['STR1_y']<=k['STR1_x']+k['STR1M'])&(k['STR1_y']>=k['STR1_x']-k['STR1M']),1,0)
#     k['STR2']=np.where((k['STR2_y']<=k['STR2_x']+k['STR2M'])&(k['STR2_y']>=k['STR2_x']-k['STR2M']),1,0)
#     k['STR']=np.where((k['STR1']==1)&(k['STR2']==1),1,0)
#     sum(k['STR'])/len(k)

#     k=tp.groupby(['CT','PREDHHSTR'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True)
#     k=pd.merge(hhstr.melt(id_vars=['CT'],value_vars=['STR1','STR2'],var_name='HHSTR',value_name='TOTAL'),k,how='inner',left_on=['CT','HHSTR'],right_on=['CT','PREDHHSTR'])
#     p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
#     p.show()
        
#     pumsszincstr=pd.DataFrame(ipfn.ipfn.product(tp['HHSIZE'].unique(),tp['HHINC'].unique(),tp['PREDHHSTR'].unique()))
#     pumsszincstr.columns=['HHSIZE','HHINC','HHSTR']
#     pumsszincstrtp=pums[pums['PUMA']==i].reset_index(drop=True)
#     pumsszincstrtp=pumsszincstrtp.groupby(['HHSIZE','HHINC','HHSTR'],as_index=False).agg({'WGTP':'sum'}).reset_index(drop=True)
#     pumsszincstr=pd.merge(pumsszincstr,pumsszincstrtp,how='left',on=['HHSIZE','HHINC','HHSTR'])
#     pumsszincstr['TOTAL']=pumsszincstr['WGTP'].fillna(0)
#     pumsszincstr=pumsszincstr[['HHSIZE','HHINC','HHSTR','TOTAL']].reset_index(drop=True)
#     k=tp.groupby(['HHSIZE','HHINC','PREDHHSTR'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True)
#     k=pd.merge(pumsszincstr,k,how='inner',left_on=['HHSIZE','HHINC','HHSTR'],right_on=['HHSIZE','HHINC','PREDHHSTR'])
#     p=px.scatter(k,x='TOTAL_x',y='TOTAL_y',hover_data=['HHSIZE','HHINC','HHSTR'])
#     p.show()
    
    
    
#     tp.to_csv(path+'POP/test1.csv',index=False)



#     # k=pd.merge(hhsize,df.groupby(['CT','HHSIZE'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['CT','HHSIZE'])
#     # k['ERR2']=np.square(k['TOTAL_y']-k['TOTAL_x'])
#     # print(np.sqrt(sum(k['ERR2'])/len(k)))
#     # p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
#     # p.show()
    
#     # k=pd.merge(hhinc,df.groupby(['CT','HHINC'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['CT','HHINC'])
#     # k['ERR2']=np.square(k['TOTAL_y']-k['TOTAL_x'])
#     # print(np.sqrt(sum(k['ERR2'])/len(k)))
#     # p=px.scatter(k,x='TOTAL_x',y='TOTAL_y')
#     # p.show()

#     # k=pd.merge(pumsszic,df.groupby(['HHSIZE','HHINC'],as_index=False).agg({'TOTAL':'sum'}).reset_index(drop=True),how='inner',on=['HHSIZE','HHINC'])
#     # k['ERR2']=np.square(k['TOTAL']-k['WGTP'])
#     # print(np.sqrt(sum(k['ERR2'])/len(k)))
#     # p=px.scatter(k,x='WGTP',y='TOTAL')
#     # p.show()