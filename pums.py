import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'



bpm=['36005','36047','36061','36081','36085','36059','36103','36119','36087','36079','36071','36027',
     '09001','09009','34017','34003','34031','34013','34039','34027','34035','34023','34025','34029',
     '34037','34041','34019','34021']
geoxwalk=pd.read_csv(path+'POP/GEOIDCROSSWALK.csv',dtype=str)



# Clean up PUMS Household
pumshh=[]
for i in ['09','34','36']:
    pumshh+=[pd.read_csv(path+'PUMS/psam_h'+i+'.csv',dtype=str)]
pumshh=pd.concat(pumshh,axis=0,ignore_index=True)
pumshh['HHID']=pumshh['SERIALNO'].copy()
pumshh['PUMA']=pumshh['ST']+pumshh['PUMA']
pumshh=pd.merge(pumshh,geoxwalk[['PUMA2010','StateCounty']].drop_duplicates(keep='first'),how='left',left_on='PUMA',right_on='PUMA2010')
pumshh=pumshh[np.isin(pumshh['StateCounty'],bpm)].reset_index(drop=True)
pumshhgq=pumshh.loc[pumshh['TYPE']!='1',['HHID','PUMA']].reset_index(drop=True)
pumshhgq=pumshhgq.drop_duplicates(keep='first').reset_index(drop=True)
pumshhgq.to_csv(path+'PUMS/pumshhgq.csv',index=False)
pumshh=pumshh[pumshh['TYPE']=='1'].reset_index(drop=True)
pumshh['HHSIZE']=np.where(pumshh['NP']=='0','VAC',
                 np.where(pumshh['NP']=='1','SIZE1',
                 np.where(pumshh['NP']=='2','SIZE2',
                 np.where(pumshh['NP']=='3','SIZE3','SIZE4'))))
pumshh['HHTYPE']=np.where(pumshh['NP']=='0','VAC',
                 np.where(pumshh['HHT2']=='02','TYPE1',
                 np.where(pumshh['HHT2']=='04','TYPE1',
                 np.where(pumshh['HHT2']=='01','TYPE2',
                 np.where(pumshh['HHT2']=='03','TYPE2',
                 np.where(pumshh['HHT2']=='05','TYPE3',
                 np.where(pumshh['HHT2']=='09','TYPE3',
                 np.where(pumshh['HHT2']=='06','TYPE4',
                 np.where(pumshh['HHT2']=='10','TYPE4','TYPE5')))))))))
pumshh['HHINC']=pd.to_numeric(pumshh['HINCP'])
pumshh['HHINC']=np.where(pumshh['NP']=='0','VAC',
                np.where(pumshh['HHINC']<5000,'INC01',
                np.where(pumshh['HHINC']<10000,'INC02',
                np.where(pumshh['HHINC']<15000,'INC03',
                np.where(pumshh['HHINC']<20000,'INC04',
                np.where(pumshh['HHINC']<25000,'INC05',
                np.where(pumshh['HHINC']<35000,'INC06',
                np.where(pumshh['HHINC']<50000,'INC07',
                np.where(pumshh['HHINC']<75000,'INC08',
                np.where(pumshh['HHINC']<100000,'INC09',
                np.where(pumshh['HHINC']<150000,'INC10','INC11')))))))))))
pumshh['HHTEN']=np.where(pumshh['NP']=='0','VAC',
                np.where(pumshh['TEN']=='3','RENTER','OWNER'))
pumshh['HHSTR']=np.where(pumshh['NP']=='0','VAC',
                np.where(pumshh['BLD']=='02','STR1D',
                np.where(pumshh['BLD']=='03','STR1A',
                np.where(pumshh['BLD']=='04','STR2',
                np.where(pumshh['BLD']=='05','STR34',
                np.where(pumshh['BLD']=='06','STR59',
                np.where(pumshh['BLD']=='07','STR10',
                np.where(pumshh['BLD']=='08','STR10',
                np.where(pumshh['BLD']=='09','STR10','STRO')))))))))
pumshh['HHBLT']=np.where(pumshh['YBL']=='01','B39',
                np.where(pumshh['YBL']=='02','B40',
                np.where(pumshh['YBL']=='03','B50',
                np.where(pumshh['YBL']=='04','B60',
                np.where(pumshh['YBL']=='05','B70',
                np.where(pumshh['YBL']=='06','B80',
                np.where(pumshh['YBL']=='07','B90',
                np.where(pumshh['YBL']=='08','B00',
                np.where(pumshh['YBL']=='09','B00',
                np.where(pumshh['YBL']=='10','B00',
                np.where(pumshh['YBL']=='11','B00',
                np.where(pumshh['YBL']=='12','B00',
                np.where(pumshh['YBL']=='13','B00',
                np.where(pumshh['YBL']=='14','B10',
                np.where(pumshh['YBL']=='15','B10',
                np.where(pumshh['YBL']=='16','B10',
                np.where(pumshh['YBL']=='17','B10','B14')))))))))))))))))
pumshh['HHBED']=np.where(pumshh['BDSP']=='0','BED0',
                np.where(pumshh['BDSP']=='1','BED1',
                np.where(pumshh['BDSP']=='2','BED2',
                np.where(pumshh['BDSP']=='3','BED3',
                np.where(pumshh['BDSP']=='4','BED4','BED5')))))
pumshh['HHVEH']=np.where(pumshh['NP']=='0','VAC',
                np.where(pumshh['VEH']=='0','VEH0',
                np.where(pumshh['VEH']=='1','VEH1',
                np.where(pumshh['VEH']=='2','VEH2',
                np.where(pumshh['VEH']=='3','VEH3','VEH4')))))
pumshh=pumshh[['HHID','PUMA','WGTP','HHSIZE','HHTYPE','HHINC','HHTEN','HHSTR','HHBLT','HHBED','HHVEH']].reset_index(drop=True)
pumshh=pumshh.drop_duplicates(keep='first').reset_index(drop=True)
pumshh.to_csv(path+'PUMS/pumshh.csv',index=False)



# Clean up PUMS Person
pumspp=[]
for i in ['09','34','36']:
    pumspp+=[pd.read_csv(path+'PUMS/psam_p'+i+'.csv',dtype=str)]
pumspp=pd.concat(pumspp,axis=0,ignore_index=True)
pumspp['PPID']=pumspp['SERIALNO']+'|'+pumspp['SPORDER']
pumspp['HHID']=pumspp['SERIALNO'].copy()
pumspp['PUMA']=pumspp['ST']+pumspp['PUMA']
pumspp=pd.merge(pumspp,geoxwalk[['PUMA2010','StateCounty']].drop_duplicates(keep='first'),how='left',left_on='PUMA',right_on='PUMA2010')
pumspp=pumspp[np.isin(pumspp['StateCounty'],bpm)].reset_index(drop=True)
pumspp['PPSEX']=np.where(pumspp['SEX']=='1','MALE','FEMALE')
pumspp['PPAGE']=pd.to_numeric(pumspp['AGEP'])
pumspp['PPAGE']=np.where(pumspp['PPAGE']<=5,'AGE01',
                np.where(pumspp['PPAGE']<=9,'AGE02',
                np.where(pumspp['PPAGE']<=14,'AGE03',
                np.where(pumspp['PPAGE']<=19,'AGE04',
                np.where(pumspp['PPAGE']<=24,'AGE05',
                np.where(pumspp['PPAGE']<=29,'AGE06',
                np.where(pumspp['PPAGE']<=34,'AGE07',
                np.where(pumspp['PPAGE']<=39,'AGE08',
                np.where(pumspp['PPAGE']<=44,'AGE09',
                np.where(pumspp['PPAGE']<=49,'AGE10',
                np.where(pumspp['PPAGE']<=54,'AGE11',
                np.where(pumspp['PPAGE']<=59,'AGE12',
                np.where(pumspp['PPAGE']<=64,'AGE13',
                np.where(pumspp['PPAGE']<=69,'AGE14',
                np.where(pumspp['PPAGE']<=74,'AGE15',
                np.where(pumspp['PPAGE']<=79,'AGE16',
                np.where(pumspp['PPAGE']<=84,'AGE17','AGE18')))))))))))))))))
pumspp['PPSCH']=np.where(pumspp['SCHG']=='01','PR',
                np.where(pumspp['SCHG']=='02','KG',
                np.where(pumspp['SCHG']=='03','G14',
                np.where(pumspp['SCHG']=='04','G14',
                np.where(pumspp['SCHG']=='05','G14',
                np.where(pumspp['SCHG']=='06','G14',
                np.where(pumspp['SCHG']=='07','G59',
                np.where(pumspp['SCHG']=='08','G59',
                np.where(pumspp['SCHG']=='09','G59',
                np.where(pumspp['SCHG']=='10','G59',
                np.where(pumspp['SCHG']=='11','HS',
                np.where(pumspp['SCHG']=='12','HS',
                np.where(pumspp['SCHG']=='13','HS',
                np.where(pumspp['SCHG']=='14','HS',
                np.where(pumspp['SCHG']=='15','CL',
                np.where(pumspp['SCHG']=='16','GS','NS'))))))))))))))))
pumspp['PPMODE']=np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='1'),'DA',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='2'),'CP2',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='3'),'CP3',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='4'),'CP4',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='5'),'CP56',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='6'),'CP56',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='7'),'CP7',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='8'),'CP7',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='9'),'CP7',
                 np.where((pumspp['JWTRNS']=='01')&(pumspp['JWRIP']=='10'),'CP7',
                 np.where(pumspp['JWTRNS']=='02','BS',
                 np.where(pumspp['JWTRNS']=='03','SW',
                 np.where(pumspp['JWTRNS']=='04','CR',
                 np.where(pumspp['JWTRNS']=='05','LR',
                 np.where(pumspp['JWTRNS']=='06','FB',
                 np.where(pumspp['JWTRNS']=='07','TC',
                 np.where(pumspp['JWTRNS']=='08','MC',
                 np.where(pumspp['JWTRNS']=='09','BC',
                 np.where(pumspp['JWTRNS']=='10','WK',
                 np.where(pumspp['JWTRNS']=='11','HM',
                 np.where(pumspp['JWTRNS']=='12','OT','NW')))))))))))))))))))))
pumsppgq=pumspp[np.isin(pumspp['RELSHIPP'],['37','38'])].reset_index(drop=True)
pumsppgq=pumsppgq[['PPID','HHID','PUMA','PWGTP','PPSEX','PPAGE','PPSCH','PPMODE']].reset_index(drop=True)
pumsppgq=pumsppgq.drop_duplicates(keep='first').reset_index(drop=True)
pumsppgq.to_csv(path+'PUMS/pumsppgq.csv',index=False)
pumspp=pumspp[~np.isin(pumspp['RELSHIPP'],['37','38'])].reset_index(drop=True)
pumspp=pumspp[['PPID','HHID','PUMA','PWGTP','PPSEX','PPAGE','PPSCH','PPMODE']].reset_index(drop=True)
pumspp=pumspp.drop_duplicates(keep='first').reset_index(drop=True)
pumspp.to_csv(path+'PUMS/pumspp.csv',index=False)




#class of worker:cow
#Travel time to work:JWMNP

#education
#employment status
#industry
#race










































# Backup

# pums=pd.read_csv(path+'POP/psam_h36.csv',dtype=str)
# pums['PUMA']=pums['ST']+pums['PUMA']
# pums=pd.merge(pums,geoxwalk[['PUMA2010','StateCounty']].drop_duplicates(keep='first'),how='left',left_on='PUMA',right_on='PUMA2010')
# pums=pums[np.isin(pums['StateCounty'],bpm)].reset_index(drop=True)



# # i='3603805'
# # df=pums[pums['PUMA']==i].reset_index(drop=True)

# df=pums.copy()
# df=df[df['TYPE']=='1'].reset_index(drop=True)
# df=df[df['NP']!='0'].reset_index(drop=True)
# df=df[df['WGTP']!='0'].reset_index(drop=True)
# df['WGTP']=pd.to_numeric(df['WGTP'])
# df['HHSIZE']=np.where(df['NP']=='1','SIZE1',
#              np.where(df['NP']=='2','SIZE2',
#              np.where(df['NP']=='3','SIZE3','SIZE4')))
# df['BEDROOM']=np.where(df['BDSP']=='0','BEDROOM1',
#               np.where(df['BDSP']=='1','BEDROOM1',
#               np.where(df['BDSP']=='2','BEDROOM2',
#               np.where(df['BDSP']=='3','BEDROOM3','BEDROOM4'))))
# df['STRUCTURE']=np.where(df['BLD']=='02','STRUCTURE1',
#                 np.where(df['BLD']=='03','STRUCTURE1','STRUCTURE2'))
# df['TENURE']=np.where(df['TEN']=='1','OWNED',
#              np.where(df['TEN']=='2','OWNED',
#              np.where(df['TEN']=='4','OWNED','RENTED')))
# df['VEHICLE']=np.where(df['VEH']=='0','VEHICLE0',
#               np.where(df['VEH']=='1','VEHICLE1',
#               np.where(df['VEH']=='2','VEHICLE2','VEHICLE3')))
# df['HHTYPE']=np.where(df['HHT2']=='02','TYPE1',
#              np.where(df['HHT2']=='04','TYPE1',
#              np.where(df['HHT2']=='01','TYPE2',
#              np.where(df['HHT2']=='03','TYPE2',
#              np.where(df['HHT2']=='05','TYPE3',
#              np.where(df['HHT2']=='09','TYPE3','TYPE4'))))))
# df['HHINC']=pd.to_numeric(df['HINCP'])
# df['HHINC']=np.where(df['HHINC']<35000,'INC1',
#             np.where(df['HHINC']<75000,'INC2',
#             np.where(df['HHINC']<150000,'INC3','INC4')))
# df=df[['WGTP','HHSIZE','HHTYPE','HHINC','TENURE','STRUCTURE','BEDROOM','VEHICLE']].reset_index(drop=True)



# tp=pd.DataFrame(ipfn.ipfn.product(['HHSIZE','HHTYPE','HHINC','TENURE','STRUCTURE','BEDROOM','VEHICLE'],['HHSIZE','HHTYPE','HHINC','TENURE','STRUCTURE','BEDROOM','VEHICLE']),columns=['DEPENDENT','FACTOR'])
# tp=tp[tp['DEPENDENT']!=tp['FACTOR']].reset_index(drop=True)
# tp['WITHOUT']=np.nan
# tp['WITH']=np.nan

# for i in tp.index:
#     tp1=pd.concat([df[[tp.loc[i,'DEPENDENT'],'WGTP']],pd.get_dummies(df.drop([tp.loc[i,'DEPENDENT'],'WGTP',tp.loc[i,'FACTOR']],axis=1),drop_first=True)],axis=1,ignore_index=False)
#     tp1x=tp1.drop([tp.loc[i,'DEPENDENT'],'WGTP'],axis=1).reset_index(drop=True)
#     tp1y=tp1[tp.loc[i,'DEPENDENT']].reset_index(drop=True)
#     tp1wt=tp1['WGTP'].reset_index(drop=True)
#     tp1reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(tp1x,tp1y,tp1wt)
#     tp.loc[i,'WITHOUT']=tp1reg.score(tp1x,tp1y,tp1wt)
    
#     tp2=pd.concat([df[[tp.loc[i,'DEPENDENT'],'WGTP']],pd.get_dummies(df.drop([tp.loc[i,'DEPENDENT'],'WGTP'],axis=1),drop_first=True)],axis=1,ignore_index=False)
#     tp2x=tp2.drop([tp.loc[i,'DEPENDENT'],'WGTP'],axis=1).reset_index(drop=True)
#     tp2y=tp2[tp.loc[i,'DEPENDENT']].reset_index(drop=True)
#     tp2wt=tp2['WGTP'].reset_index(drop=True)
#     tp2reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(tp2x,tp2y,tp2wt)
#     tp.loc[i,'WITH']=tp2reg.score(tp2x,tp2y,tp2wt)

# tp['DIFF']=tp['WITH']-tp['WITHOUT']
# tp.to_csv(path+'tp.csv',index=False)


# k=pd.concat([df[['VEHICLE']],pd.get_dummies(df[['STRUCTURE','HHSIZE']],drop_first=True)],axis=1,ignore_index=False)
# kx=k.drop(['VEHICLE'],axis=1).reset_index(drop=True)
# ky=k['VEHICLE'].reset_index(drop=True)
# reg=sklearn.linear_model.LogisticRegression(max_iter=1000).fit(kx,ky)
# reg.score(kx,ky)
# sm.MNLogit(ky,sm.add_constant(kx)).fit(maxiter=1000).summary()








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
    

    










