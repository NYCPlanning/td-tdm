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



rc=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/RemoteComplex.csv',dtype=str,converters={'CplxID':float,'CplxLat':float,'CplxLong':float,'Hub':float})
df=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/OUTPUT/dfunitentry.csv',dtype=str,converters={'entries':float,'gooducs':float,'flagtime':float,'flagentry':float})
df=df.groupby(['unit','firstdate'],as_index=False).agg({'entries':'sum'}).reset_index(drop=True)
df['wkd']=[datetime.datetime.strptime(x,'%m/%d/%Y').weekday() for x in df['firstdate']]
df=df[np.isin(df['wkd'],[0,1,2,3,4])].reset_index(drop=True)
df['year']=[x[6:] for x in df['firstdate']]
df=df[df['year']=='2019'].reset_index(drop=True)
df=df.groupby(['unit'],as_index=False).agg({'entries':'mean'}).reset_index(drop=True)
df.columns=['Remote','Entries']
df=pd.merge(df,rc,how='left',on='Remote')
df=df.groupby(['CplxID'],as_index=False).agg({'Entries':'sum'}).reset_index(drop=True)
df=pd.merge(rc.drop('Remote',axis=1).drop_duplicates(keep='first').reset_index(drop=True),df,how='left',on='CplxID')
df=df[['CplxID','CplxLat','CplxLong','Entries']].reset_index(drop=True)
df=gpd.GeoDataFrame(df,geometry=[shapely.geometry.Point(x,y) for x,y in zip(df['CplxLong'],df['CplxLat'])],crs=4326)
df=df.to_crs(6539)
df['geometry']=df.buffer(2640)
nta=gpd.read_file('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/ntaclippedadj.shp')
nta.crs=4326
nta=nta.to_crs(6539)
df=gpd.overlay(df,nta,how='intersection')
df['area']=[x.area for x in df['geometry']]
dfsum=df.groupby(['CplxID'],as_index=False).agg({'area':'sum'}).reset_index(drop=True)
df=pd.merge(df,dfsum,how='inner',on='CplxID')
df['pct']=df['area_x']/df['area_y']
df=df[['CplxID','NTACode','pct','Entries']].reset_index(drop=True)
df['Entries']=df['Entries']*df['pct']
df=df.groupby(['NTACode'],as_index=False).agg({'Entries':'sum'}).reset_index(drop=True)
df=pd.merge(nta,df,how='inner',on='NTACode')
df=df.to_crs(4326)
df.to_file('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/TURNSTILE/turnstile.shp')









rc=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/RemoteComplex.csv',dtype=str,converters={'CplxID':float,'CplxLat':float,'CplxLong':float,'Hub':float})
df=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/OUTPUT/dfunitentry.csv',dtype=str,converters={'entries':float,'gooducs':float,'flagtime':float,'flagentry':float})
df=df.groupby(['unit','firstdate'],as_index=False).agg({'entries':'sum'}).reset_index(drop=True)
df['wkd']=[datetime.datetime.strptime(x,'%m/%d/%Y').weekday() for x in df['firstdate']]
df=df[np.isin(df['wkd'],[0,1,2,3,4])].reset_index(drop=True)
df['month']=[x.split('/')[0] for x in df['firstdate']]
df['year']=[x.split('/')[-1] for x in df['firstdate']]
dfpre=df[df['month']=='04'].reset_index(drop=True)
dfpre=dfpre[dfpre['year']=='2019'].reset_index(drop=True)
dfpre=dfpre.groupby(['unit'],as_index=False).agg({'entries':'mean'}).reset_index(drop=True)
dfpre.columns=['Remote','E201904']
dfpost=df[df['month']=='04'].reset_index(drop=True)
dfpost=dfpost[dfpost['year']=='2020'].reset_index(drop=True)
dfpost=dfpost.groupby(['unit'],as_index=False).agg({'entries':'mean'}).reset_index(drop=True)
dfpost.columns=['Remote','E202004']
df=pd.merge(dfpre,dfpost,how='inner',on='Remote')
df=pd.merge(df,rc,how='left',on='Remote')
df=df.groupby(['CplxID'],as_index=False).agg({'E201904':'sum','E202004':'sum'}).reset_index(drop=True)
df=pd.merge(rc.drop('Remote',axis=1).drop_duplicates(keep='first').reset_index(drop=True),df,how='left',on='CplxID')
df=df[['CplxID','CplxLat','CplxLong','E201904','E202004']].reset_index(drop=True)
df=gpd.GeoDataFrame(df,geometry=[shapely.geometry.Point(x,y) for x,y in zip(df['CplxLong'],df['CplxLat'])],crs=4326)
df=df.to_crs(6539)
df['geometry']=df.buffer(2640)
nta=gpd.read_file('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/ntaclippedadj.shp')
nta.crs=4326
nta=nta.to_crs(6539)
df=gpd.overlay(df,nta,how='intersection')
df['area']=[x.area for x in df['geometry']]
dfsum=df.groupby(['CplxID'],as_index=False).agg({'area':'sum'}).reset_index(drop=True)
df=pd.merge(df,dfsum,how='inner',on='CplxID')
df['pct']=df['area_x']/df['area_y']
df=df[['CplxID','NTACode','pct','E201904','E202004']].reset_index(drop=True)
df['E201904']=df['E201904']*df['pct']
df['E202004']=df['E202004']*df['pct']
df=df.groupby(['NTACode'],as_index=False).agg({'E201904':'sum','E202004':'sum'}).reset_index(drop=True)
df=pd.merge(nta,df,how='inner',on='NTACode')
df=df.to_crs(4326)
df=df.drop('geometry',axis=1)
df.to_csv('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/TURNSTILE/turnstile2.csv',index=False)





