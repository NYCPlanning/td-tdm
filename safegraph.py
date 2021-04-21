import s3fs
import pandas as pd
import numpy as np
import sqlalchemy as sal
import geopandas as gpd
import shapely
import json



pd.set_option('display.max_columns', None)



df=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SAFEGRAPH/20190809.csv',dtype=str,converters={'D20190809':float})
df=df[np.logical_or([x[0:5] in ['36005','36047','36061','36081','36085'] for x in df['origin']],
                    [x[0:5] in ['36005','36047','36061','36081','36085'] for x in df['destination']])].reset_index(drop=True)
df=df.sort_values('D20190809',ascending=False).reset_index(drop=True)
nta=gpd.read_file('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/ntaclippedadj.shp')
nta.crs=4326

