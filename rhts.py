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


rhts=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/RHTS/LINKED_Public.csv',dtype=str)
rhts['pid']=rhts['SAMPN']+'|'+rhts['PERNO']

df=[]
for i in rhts['pid'].unique():
    tp=rhts[rhts['pid']==i].reset_index(drop=True)
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
    tp['wt']=pd.to_numeric(list(rhts.loc[rhts['pid']==i,'HH_WHT2'])[0])
    tp=tp[['pid','wt','hour','dest']].reset_index(drop=True)
    df+=[tp]
df=pd.concat(df,axis=0,ignore_index=True)
df=df.groupby(['dest','hour'],as_index=False).agg({'wt':'sum'}).reset_index(drop=True)
df=df.pivot(index='dest',columns='hour',values='wt').reset_index(drop=False)
df.to_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhts.csv',index=False)


df=pd.read_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-tdm/rhts.csv')
df['tractid']=[str(x).zfill(11) for x in df['dest']]
df['county']=[x[0:5] for x in df['tractid']]
df=df[np.isin(df['county'],['36005','36047','36061','36081','36085'])].reset_index(drop=True)
df=df.fillna(0)
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




