import requests
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
apikey=pd.read_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-acsapi/apikey.csv',header=None).loc[0,0]



nyc=['36005','36047','36061','36081','36085']
nymtc=['36005','36047','36061','36081','36085','36059','36103','36119','36087','36079']
bpm=['36005','36047','36061','36081','36085','36059','36103','36119','36087','36079','36071','36027',
     '09001','09009','34017','34003','34031','34013','34039','34027','34035','34023','34025','34029',
     '34037','34041','34019','34021']



# Household Total
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B25002)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B25002_001E','B25002_001M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/hhtt.csv',index=False)



# Household Occupancy
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B25002)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B25002_001E','B25002_001M','B25002_002E','B25002_002M',
           'B25002_003E','B25002_003M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','OCC','OCCM','VAC','VACM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/hhocc.csv',index=False)



# Household Size
hhocc=pd.read_csv(path+'ACS/hhocc.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S2501)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S2501_C01_002E','S2501_C01_002M','S2501_C01_003E','S2501_C01_003M',
           'S2501_C01_004E','S2501_C01_004M','S2501_C01_005E','S2501_C01_005M']].reset_index(drop=True)
    rs.columns=['CT','SIZE1','SIZE1M','SIZE2','SIZE2M','SIZE3','SIZE3M','SIZE4','SIZE4M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(df,hhocc,how='inner',on='CT')
df=df[['CT','TT','TTM','VAC','VACM','SIZE1','SIZE1M','SIZE2','SIZE2M','SIZE3','SIZE3M',
       'SIZE4','SIZE4M']].reset_index(drop=True)
df.to_csv(path+'ACS/hhsize.csv',index=False)



# Household Type
hhocc=pd.read_csv(path+'ACS/hhocc.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/profile?get=NAME,group(DP02)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','DP02_0002E','DP02_0002M','DP02_0003E','DP02_0003M',
           'DP02_0004E','DP02_0004M','DP02_0005E','DP02_0005M',
           'DP02_0006E','DP02_0006M','DP02_0007E','DP02_0007M',
           'DP02_0008E','DP02_0008M','DP02_0010E','DP02_0010M',
           'DP02_0011E','DP02_0011M','DP02_0012E','DP02_0012M']].reset_index(drop=True)
    rs.columns=['CT','MC','MCM','MCC','MCCM','CC','CCM','CCC','CCCM','MH','MHM','MHC','MHCM',
                'MHA','MHAM','FH','FHM','FHC','FHCM','FHA','FHAM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(df,hhocc,how='inner',on='CT')
df=df[['CT','TT','TTM','VAC','VACM','MC','MCM','MCC','MCCM','CC','CCM','CCC','CCCM','MH','MHM',
       'MHC','MHCM','MHA','MHAM','FH','FHM','FHC','FHCM','FHA','FHAM']].reset_index(drop=True)
df.to_csv(path+'ACS/hhtype.csv',index=False)



# Household Income
hhocc=pd.read_csv(path+'ACS/hhocc.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S2503)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S2503_C01_002E','S2503_C01_002M','S2503_C01_003E','S2503_C01_003M',
           'S2503_C01_004E','S2503_C01_004M','S2503_C01_005E','S2503_C01_005M',
           'S2503_C01_006E','S2503_C01_006M','S2503_C01_007E','S2503_C01_007M',
           'S2503_C01_008E','S2503_C01_008M','S2503_C01_009E','S2503_C01_009M',
           'S2503_C01_010E','S2503_C01_010M','S2503_C01_011E','S2503_C01_011M',
           'S2503_C01_012E','S2503_C01_012M']].reset_index(drop=True)
    rs.columns=['CT','INC01','INC01M','INC02','INC02M','INC03','INC03M','INC04','INC04M',
                'INC05','INC05M','INC06','INC06M','INC07','INC07M','INC08','INC08M','INC09','INC09M',
                'INC10','INC10M','INC11','INC11M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(df,hhocc,how='inner',on='CT')
df=df[['CT','TT','TTM','VAC','VACM','INC01','INC01M','INC02','INC02M','INC03','INC03M','INC04','INC04M',
       'INC05','INC05M','INC06','INC06M','INC07','INC07M','INC08','INC08M','INC09','INC09M',
       'INC10','INC10M','INC11','INC11M']].reset_index(drop=True)
df.to_csv(path+'ACS/hhinc.csv',index=False)



# Household Tenure
hhocc=pd.read_csv(path+'ACS/hhocc.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B25003)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B25003_002E','B25003_002M','B25003_003E','B25003_003M']].reset_index(drop=True)
    rs.columns=['CT','OWNER','OWNERM','RENTER','RENTERM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(df,hhocc,how='inner',on='CT')
df=df[['CT','TT','TTM','VAC','VACM','OWNER','OWNERM','RENTER','RENTERM']].reset_index(drop=True)
df.to_csv(path+'ACS/hhten.csv',index=False)



# Household Structure
hhocc=pd.read_csv(path+'ACS/hhocc.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S2504)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S2504_C01_002E','S2504_C01_002M','S2504_C01_003E','S2504_C01_003M',
           'S2504_C01_004E','S2504_C01_004M','S2504_C01_005E','S2504_C01_005M',
           'S2504_C01_006E','S2504_C01_006M','S2504_C01_007E','S2504_C01_007M',
           'S2504_C01_008E','S2504_C01_008M']].reset_index(drop=True)
    rs.columns=['CT','STR1D','STR1DM','STR1A','STR1AM','STR2','STR2M','STR34','STR34M',
                'STR59','STR59M','STR10','STR10M','STRO','STROM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(df,hhocc,how='inner',on='CT')
df=df[['CT','TT','TTM','VAC','VACM','STR1D','STR1DM','STR1A','STR1AM','STR2','STR2M','STR34','STR34M',
       'STR59','STR59M','STR10','STR10M','STRO','STROM']].reset_index(drop=True)
df.to_csv(path+'ACS/hhstr.csv',index=False)



# Household Built
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B25034)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B25034_001E','B25034_001M','B25034_002E','B25034_002M',
           'B25034_003E','B25034_003M','B25034_004E','B25034_004M',
           'B25034_005E','B25034_005M','B25034_006E','B25034_006M',
           'B25034_007E','B25034_007M','B25034_008E','B25034_008M',
           'B25034_009E','B25034_009M','B25034_010E','B25034_010M',
           'B25034_011E','B25034_011M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','B14','B14M','B10','B10M','B00','B00M','B90','B90M','B80','B80M',
                'B70','B70M','B60','B60M','B50','B50M','B40','B40M','B39','B39M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/hhblt.csv',index=False)



# Household Bedroom
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B25041)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B25041_001E','B25041_001M','B25041_002E','B25041_002M',
           'B25041_003E','B25041_003M','B25041_004E','B25041_004M',
           'B25041_005E','B25041_005M','B25041_006E','B25041_006M',
           'B25041_007E','B25041_007M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','BED0','BED0M','BED1','BED1M','BED2','BED2M','BED3','BED3M',
                'BED4','BED4M','BED5','BED5M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/hhbed.csv',index=False)



# Household Vehicles
hhocc=pd.read_csv(path+'ACS/hhocc.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B08203)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B08203_002E','B08203_002M','B08203_003E','B08203_003M',
           'B08203_004E','B08203_004M','B08203_005E','B08203_005M',
           'B08203_006E','B08203_006M']].reset_index(drop=True)
    rs.columns=['CT','VEH0','VEH0M','VEH1','VEH1M','VEH2','VEH2M','VEH3','VEH3M','VEH4','VEH4M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(df,hhocc,how='inner',on='CT')
df=df[['CT','TT','TTM','VAC','VACM','VEH0','VEH0M','VEH1','VEH1M','VEH2','VEH2M','VEH3','VEH3M',
       'VEH4','VEH4M']].reset_index(drop=True)
df.to_csv(path+'ACS/hhveh.csv',index=False)



# Group Quarter Population
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B26001)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B26001_001E','B26001_001M']].reset_index(drop=True)
    rs.columns=['CT','GQTT','GQTTM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/gqtt.csv',index=False)



# Total Population
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S0101)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S0101_C01_001E','S0101_C01_001M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/pptt.csv',index=False)



# Sex
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S0101)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S0101_C01_001E','S0101_C01_001M','S0101_C03_001E','S0101_C03_001M',
           'S0101_C05_001E','S0101_C05_001M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','MALE','MALEM','FEMALE','FEMALEM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/ppsex.csv',index=False)



# Age
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S0101)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S0101_C01_001E','S0101_C01_001M','S0101_C01_002E','S0101_C01_002M',
           'S0101_C01_003E','S0101_C01_003M','S0101_C01_004E','S0101_C01_004M',
           'S0101_C01_005E','S0101_C01_005M','S0101_C01_006E','S0101_C01_006M',
           'S0101_C01_007E','S0101_C01_007M','S0101_C01_008E','S0101_C01_008M',
           'S0101_C01_009E','S0101_C01_009M','S0101_C01_010E','S0101_C01_010M',
           'S0101_C01_011E','S0101_C01_011M','S0101_C01_012E','S0101_C01_012M',
           'S0101_C01_013E','S0101_C01_013M','S0101_C01_014E','S0101_C01_014M',
           'S0101_C01_015E','S0101_C01_015M','S0101_C01_016E','S0101_C01_016M',
           'S0101_C01_017E','S0101_C01_017M','S0101_C01_018E','S0101_C01_018M',
           'S0101_C01_019E','S0101_C01_019M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','AGE01','AGE01M','AGE02','AGE02M','AGE03','AGE03M','AGE04','AGE04M',
                'AGE05','AGE05M','AGE06','AGE06M','AGE07','AGE07M','AGE08','AGE08M','AGE09','AGE09M',
                'AGE10','AGE10M','AGE11','AGE11M','AGE12','AGE12M','AGE13','AGE13M','AGE14','AGE14M',
                'AGE15','AGE15M','AGE16','AGE16M','AGE17','AGE17M','AGE18','AGE18M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/ppage.csv',index=False)



# School
pptt=pd.read_csv(path+'ACS/pptt.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S1401)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S1401_C01_001E','S1401_C01_001M','S1401_C01_002E','S1401_C01_002M',
           'S1401_C01_004E','S1401_C01_004M','S1401_C01_005E','S1401_C01_005M',
           'S1401_C01_006E','S1401_C01_006M','S1401_C01_007E','S1401_C01_007M',
           'S1401_C01_008E','S1401_C01_008M','S1401_C01_009E','S1401_C01_009M']].reset_index(drop=True)
    rs.columns=['CT','TS','TSM','PR','PRM','KG','KGM','G14','G14M','G58','G58M','HS','HSM','CL','CLM',
                'GS','GSM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(pptt,df,how='left',on='CT')
df=df.fillna(0)
df['TT']=pd.to_numeric(df['TT'])
df['TTM']=pd.to_numeric(df['TTM'])
df['TS']=pd.to_numeric(df['TS'])
df['TSM']=pd.to_numeric(df['TSM'])
df['NS']=df['TT']-df['TS']
df['NSM']=np.sqrt(df['TTM']**2+df['TSM']**2)
df=df[['CT','TT','TTM','NS','NSM','PR','PRM','KG','KGM','G14','G14M','G58','G58M','HS','HSM','CL','CLM',
       'GS','GSM']].reset_index(drop=True)
df.to_csv(path+'ACS/ppsch.csv',index=False)



# Mode
pptt=pd.read_csv(path+'ACS/pptt.csv',dtype=str)
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B08301)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B08301_001E','B08301_001M','B08301_003E','B08301_003M',
           'B08301_005E','B08301_005M','B08301_006E','B08301_006M',
           'B08301_007E','B08301_007M','B08301_008E','B08301_008M',
           'B08301_009E','B08301_009M','B08301_011E','B08301_011M',
           'B08301_012E','B08301_012M','B08301_013E','B08301_013M',
           'B08301_014E','B08301_014M','B08301_015E','B08301_015M',
           'B08301_016E','B08301_016M','B08301_017E','B08301_017M',
           'B08301_018E','B08301_018M','B08301_019E','B08301_019M',
           'B08301_020E','B08301_020M','B08301_021E','B08301_021M']].reset_index(drop=True)
    rs.columns=['CT','TW','TWM','DA','DAM','CP2','CP2M','CP3','CP3M','CP4','CP4M','CP56','CP56M',
                'CP7','CP7M','BS','BSM','SW','SWM','CR','CRM','LR','LRM','FB','FBM','TC','TCM',
                'MC','MCM','BC','BCM','WK','WKM','OT','OTM','HM','HMM']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df=pd.merge(pptt,df,how='left',on='CT')
df=df.fillna(0)
df['TT']=pd.to_numeric(df['TT'])
df['TTM']=pd.to_numeric(df['TTM'])
df['TW']=pd.to_numeric(df['TW'])
df['TWM']=pd.to_numeric(df['TWM'])
df['NW']=df['TT']-df['TW']
df['NWM']=np.sqrt(df['TTM']**2+df['TWM']**2)
df=df[['CT','TT','TTM','NW','NWM','DA','DAM','CP2','CP2M','CP3','CP3M','CP4','CP4M','CP56','CP56M',
       'CP7','CP7M','BS','BSM','SW','SWM','CR','CRM','LR','LRM','FB','FBM','TC','TCM','MC','MCM',
       'BC','BCM','WK','WKM','OT','OTM','HM','HMM']].reset_index(drop=True)
df.to_csv(path+'ACS/ppmode.csv',index=False)







