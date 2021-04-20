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

pu=pd.read_csv('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/FHV/weekdaypunta.csv')
nta=gpd.read_file('C:/Users/mayij/Desktop/DOC/DCP2020/COVID19/SUBWAY/TURNSTILE/ntaclippedadj.shp')
nta.crs=4326
pu=pd.merge(nta,pu,how='inner',left_on='NTACode',right_on='puntaid')
pu.to_file('C:/Users/mayij/Desktop/DOC/DCP2021/TRAVEL DEMAND MODEL/FHV/weekdaypunta.shp')












