import requests
import pandas as pd

path='C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/'
apikey=pd.read_csv('C:/Users/mayij/Desktop/DOC/GITHUB/td-acsapi/apikey.csv',header=None).loc[0,0]



nyc=['36005','36047','36061','36081','36085']
df=[]
for i in nyc:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/?get=NAME,group(B08202)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','B08202_001E','B08202_006E','B08202_009E','B08202_013E','B08202_018E']].reset_index(drop=True)
    rs.columns=['CT','TT','SIZE1','SIZE2','SIZE3','SIZE4']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/hhsize.csv',index=False)



nyc=['36005','36047','36061','36081','36085']
df=[]
for i in nyc:
    rs=requests.get('https://api.census.gov/data/2019/acs/acs5/subject?get=NAME,group(S2503)&for=tract:*&in=state:'+i[:2]+' county:'+i[2:]+'&key='+apikey).json()
    rs=pd.DataFrame(rs)
    rs.columns=rs.loc[0]
    rs=rs.loc[1:].reset_index(drop=True)
    rs['geoid']=[x[9:] for x in rs['GEO_ID']]
    rs=rs[['geoid','S2503_C01_001E','S2503_C01_002E','S2503_C01_003E','S2503_C01_004E','S2503_C01_005E',
           'S2503_C01_006E','S2503_C01_007E','S2503_C01_008E','S2503_C01_009E','S2503_C01_010E',
           'S2503_C01_011E','S2503_C01_012E']].reset_index(drop=True)
    rs.columns=['CT','TT','INC01','INC02','INC03','INC04','INC05','INC06','INC07','INC08','INC09','INC10','INC11']
    df+=[rs]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv(path+'ACS/hhinc.csv',index=False)


hhworker
hhrent
hhveh


