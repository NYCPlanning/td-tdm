import urllib.request
import shutil
import os
import pandas as pd
import numpy as np
import datetime
import pytz
import geopandas as gpd
import shapely



pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'

# Map Population by Time of Day
# Trips
rhtstrip=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/RHTS/LINKED_Public.csv',dtype=str)
rhtstrip['pid']=rhtstrip['SAMPN']+'|'+rhtstrip['PERNO']
df=[]
for i in rhtstrip['pid'].unique():
    tp=rhtstrip[rhtstrip['pid']==i].reset_index(drop=True)
    tp['hour']=pd.to_numeric(tp['TRP_ARR_HR'])
    tp['min']=pd.to_numeric(tp['TRP_ARR_MIN'])
    tp=tp.sort_values(['hour','min']).reset_index(drop=True)
    tp['dest']=tp['DTRACT']
    tp=tp[['hour','dest']].reset_index(drop=True)
    tp['hour']=['h'+str(x).zfill(2) for x in tp['hour']]
    tp=tp.drop_duplicates(['hour'],keep='last').reset_index(drop=True)
    tp=pd.merge(pd.DataFrame(data=['h'+str(x).zfill(2) for x in range(0,24)],columns=['hour']),tp,how='left',on='hour')
    tp=pd.concat([tp]*2,axis=0,ignore_index=True)
    for j in tp.index[1:]:
        if pd.isna(tp.loc[j,'dest']):
            tp.loc[j,'dest']=tp.loc[j-1,'dest']
    tp=tp[pd.notna(tp['dest'])].drop_duplicates(keep='first').sort_values(['hour']).reset_index(drop=True)
    tp['pid']=i
    tp['wt']=pd.to_numeric(list(rhtstrip.loc[rhtstrip['pid']==i,'HH_WHT2'])[0])
    tp=tp[['pid','wt','hour','dest']].reset_index(drop=True)
    df+=[tp]
df=pd.concat(df,axis=0,ignore_index=True)
df=df.groupby(['dest','hour'],as_index=False).agg({'wt':'sum'}).reset_index(drop=True)
df=df.pivot(index='dest',columns='hour',values='wt').reset_index(drop=False)
df.to_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhtstrip.csv',index=False)



# Non-trips
rhtsps=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/RHTS/PER_Public.csv',dtype=str,encoding='latin-1')
rhtshh=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/RHTS/HH_Public.csv',dtype=str,encoding='latin-1')
df=pd.merge(rhtsps,rhtshh,how='inner',on='SAMPN')
df['pid']=df['SAMPN']+'|'+df['PERNO']
df['wt']=pd.to_numeric(df['HH_WHT2_x'])
df['dest']=df['HTRACT'].copy()
df=df.loc[df['PTRIPS']=='0',['pid','wt','dest']].reset_index(drop=True)
df=df.groupby(['dest'],as_index=False).agg({'wt':'sum'}).reset_index(drop=True)
for i in ['h'+str(x).zfill(2) for x in range(0,24)]:
    df[i]=df['wt'].copy()
df=df[['dest']+['h'+str(x).zfill(2) for x in range(0,24)]].reset_index(drop=True)
df.to_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhtsnontrip.csv',index=False)


# Combine
rhtstrip=pd.read_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhtstrip.csv')
rhtsnontrip=pd.read_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhtsnontrip.csv')
df=pd.merge(rhtstrip,rhtsnontrip,how='outer',on='dest')
df=df.fillna(0)
for i in ['h'+str(x).zfill(2) for x in range(0,24)]:
    df[i]=df[i+'_x']+df[i+'_y']
df['tractid']=[str(x).zfill(11) for x in df['dest']]
df['county']=[x[0:5] for x in df['tractid']]
df=df[np.isin(df['county'],['36005','36047','36061','36081','36085'])].reset_index(drop=True)
quadstatect=gpd.read_file('C:/Users/mayij/Desktop/DOC/DCP2018/TRAVELSHEDREVAMP/shp/quadstatectclipped.shp')
quadstatect.crs=4326
df=pd.merge(quadstatect,df,how='inner',on='tractid')
df=df[['tractid','h00','h01','h02','h03','h04','h05','h06','h07','h08','h09','h10','h11','h12','h13','h14','h15',
       'h16','h17','h18','h19','h20','h21','h22','h23','geometry']].reset_index(drop=True)
# df=df.melt(id_vars=['tractid','geometry'],value_vars=['h'+str(x).zfill(2) for x in range(0,24)])
# df['h17'].describe(percentiles=np.arange(0.2,1,0.2))
for i in ['h'+str(x).zfill(2) for x in range(0,24)]:
    df[i+'cat']=np.where(df[i]>10000,'> 10,000',
                np.where(df[i]>8000,'8,001 ~ 10,000',
                np.where(df[i]>6000,'6,001 ~ 8,000',
                np.where(df[i]>4000,'4,001 ~ 6,000',
                np.where(df[i]>2000,'2,001 ~ 4,000','<= 2,000')))))
df.to_file('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhts.geojson',driver='GeoJSON')





# Clean Household
rhtshh=pd.read_csv(path+'RHTS/HH_Public.csv',dtype=str)
rhtshh['HHID']=rhtshh['SAMPN'].copy()
rhtshh['WGTP']=pd.to_numeric(rhtshh['HH_WHT2'])
rhtshh['HHSIZE']=np.where(rhtshh['HHSIZ']=='1','SIZE1',
                 np.where(rhtshh['HHSIZ']=='2','SIZE2',
                 np.where(rhtshh['HHSIZ']=='3','SIZE3','SIZE4')))
rhtshh['HHTYPE']=np.where(rhtshh['HHSTRUC']=='1','TYPE1',
                 np.where(rhtshh['HHSTRUC']=='2','TYPE2',
                 np.where(rhtshh['HHSTRUC']=='3','TYPE3',
                 np.where(rhtshh['HHSTRUC']=='4','TYPE4',
                 np.where(rhtshh['HHSTRUC']=='5','TYPE5',
                 np.where(rhtshh['HHSTRUC']=='6','TYPE6','OTH'))))))
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
rhtshh=rhtshh[['HHID','WGTP','HHSIZE','HHTYPE','HHINC','HHSTR','HHVEH']].reset_index(drop=True)
rhtshh.to_csv(path+'RHTS/rhtshh.csv',index=False)



# Clean Person
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
rhtspp.to_csv(path+'RHTS/rhtspp.csv',index=False)





