import requests
import pandas as pd

pd.set_option('display.max_columns', None)
path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
apikey=pd.read_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-acsapi/apikey.csv',header=None).loc[0,0]



nyc=['36005','36047','36061','36081','36085']
nymtc=['36005','36047','36061','36081','36085','36059','36103','36119','36087','36079']
bpm=['36005','36047','36061','36081','36085','36059','36103','36119','36087','36079','36071','36027',
     '09001','09009','34017','34003','34031','34013','34039','34027','34035','34023','34025','34029',
     '34037','34041','34019','34021']



# Household Structure
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S1101)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S1101_C01_009E','S1101_C01_009M','S1101_C01_014E','S1101_C01_014M',
           'S1101_C01_015E','S1101_C01_015M','S1101_C01_016E','S1101_C01_016M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','STR1','STR1M','STR2','STR2M','STR3','STR3M']
    rs['STR1']=(rs['TT'].astype(int)*rs['STR1'].astype(float)/100).astype(int)
    rs['STR1M']=(rs['TT'].astype(int)*rs['STR1M'].astype(float)/100).astype(int)
    rs['STR2']=(rs['TT'].astype(int)*rs['STR2'].astype(float)/100).astype(int)
    rs['STR2M']=(rs['TT'].astype(int)*rs['STR2M'].astype(float)/100).astype(int)
    rs['STR3']=(rs['TT'].astype(int)*rs['STR3'].astype(float)/100).astype(int)
    rs['STR3M']=(rs['TT'].astype(int)*rs['STR3M'].astype(float)/100).astype(int)
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'POP/hhstr.csv',index=False)



# Household Size
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S2501)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S2501_C01_001E','S2501_C01_001M','S2501_C01_002E','S2501_C01_002M',
           'S2501_C01_003E','S2501_C01_003M','S2501_C01_004E','S2501_C01_004M',
           'S2501_C01_005E','S2501_C01_005M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','SIZE1','SIZE1M','SIZE2','SIZE2M','SIZE3','SIZE3M','SIZE4','SIZE4M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'POP/hhsize.csv',index=False)



# Household Income
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S2503)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S2503_C01_001E','S2503_C01_001M','S2503_C01_002E','S2503_C01_002M',
           'S2503_C01_003E','S2503_C01_003M','S2503_C01_004E','S2503_C01_004M',
           'S2503_C01_005E','S2503_C01_005M','S2503_C01_006E','S2503_C01_006M',
           'S2503_C01_007E','S2503_C01_007M','S2503_C01_008E','S2503_C01_008M',
           'S2503_C01_009E','S2503_C01_009M','S2503_C01_010E','S2503_C01_010M',
           'S2503_C01_011E','S2503_C01_011M','S2503_C01_012E','S2503_C01_012M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','INC01','INC01M','INC02','INC02M','INC03','INC03M','INC04','INC04M',
                'INC05','INC05M','INC06','INC06M','INC07','INC07M','INC08','INC08M','INC09','INC09M',
                'INC10','INC10M','INC11','INC11M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'POP/hhinc.csv',index=False)



# Household Vehicles
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B08203)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B08203_001E','B08203_001M','B08203_002E','B08203_002M','B08203_003E','B08203_003M',
           'B08203_004E','B08203_004M','B08203_005E','B08203_005M','B08203_006E','B08203_006M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','VEH0','VEH0M','VEH1','VEH1M','VEH2','VEH2M','VEH3','VEH3M',
                'VEH4','VEH4M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'POP/hhveh.csv',index=False)



# Household Workers
df=[]
for i in bpm:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B08202)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B08202_001E','B08202_001M','B08202_002E','B08202_002M','B08202_003E','B08202_003M',
           'B08202_004E','B08202_004M','B08202_005E','B08202_005M']].reset_index(drop=True)
    rs.columns=['CT','TT','TTM','WORK0','WORK0M','WORK1','WORK1M','WORK2','WORK2M','WORK3','WORK3M']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'POP/hhwork.csv',index=False)









