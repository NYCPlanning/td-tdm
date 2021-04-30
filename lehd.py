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



# Clean up RAC
rac=[]
for i in ['ct','nj','ny']:
    rac+=[pd.read_csv(path+'LEHD/'+i+'_rac_S000_JT01_2018.csv',dtype=float,converters={'h_geocode':str})]
rac=pd.concat(rac,axis=0,ignore_index=True)
rac.columns=['BK','TW','AGE301','AGE302','AGE303','EARN1','EARN2','EARN3','AGR','EXT','UTL','CON','MFG',
             'WHL','RET','TRN','INF','FIN','RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM',
             'WHT','BLK','NTV','ASN','PCF','TWO','NHS','HSP','LH','HS','SCAD','BDGD','MALE','FEMALE',
             'DATE']
rac=pd.merge(rac,geoxwalk,how='inner',left_on='BK',right_on='Block2010')
rac=rac[np.isin(rac['StateCounty'],bpm)].reset_index(drop=True)
rac=rac.groupby(['CensusTract2010'],as_index=False).agg({'TW':'sum','AGE301':'sum','AGE302':'sum',
                                                         'AGE303':'sum','AGR':'sum','EXT':'sum',
                                                         'UTL':'sum','CON':'sum','MFG':'sum','WHL':'sum',
                                                         'RET':'sum','TRN':'sum','INF':'sum','FIN':'sum',
                                                         'RER':'sum','PRF':'sum','MNG':'sum','WMS':'sum',
                                                         'EDU':'sum','MED':'sum','ENT':'sum','ACC':'sum',
                                                         'SRV':'sum','ADM':'sum','LH':'sum','HS':'sum',
                                                         'SCAD':'sum','BDGD':'sum','MALE':'sum',
                                                         'FEMALE':'sum'}).reset_index(drop=True)
rac.columns=['CT','TW','AGE301','AGE302','AGE303','AGR','EXT','UTL','CON','MFG','WHL','RET','TRN',
             'INF','FIN','RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM','LH','HS','SCAD',
             'BDGD','MALE','FEMALE']
rac.to_csv(path+'LEHD/rac.csv',index=False)

# Validation
dfpp=pd.read_csv(path+'POP/dfpp.csv',dtype=str,converters={'PWGTP':float,'TOTAL':float})
dfppwk=dfpp[(dfpp['PPMODE']!='NW')&(dfpp['PPIND']!='MIL')].reset_index(drop=True)
dfppwk['RACAGE']=np.where(np.isin(dfppwk['PPAGE'],['AGE04','AGE05','AGE06']),'AGE301',
                 np.where(np.isin(dfppwk['PPAGE'],['AGE07','AGE08','AGE09','AGE10','AGE11']),'AGE302',
                 np.where(np.isin(dfppwk['PPAGE'],['AGE12','AGE13','AGE14','AGE15','AGE16','AGE17','AGE18']),'AGE303','OTH')))
rac=pd.read_csv(path+'LEHD/rac.csv',dtype=float,converters={'CT':str})
# RAC missing off-the-book residents

# Check RACTW
k=pd.merge(dfppwk.groupby(['CT'],as_index=False).agg({'TOTAL':'sum'}),rac[['CT','TW']],how='inner',on=['CT'])
k=k.sort_values('TW').reset_index(drop=True)
k['HOVER']='CT: '+k['CT']+'<br>'+'MODEL: '+k['TOTAL'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TW'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL']-k['TW'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL'],
                               y=k['TW'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL'])],
                               y=[0,max(k['TOTAL'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'RACTW (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/ractwpt.html',include_plotlyjs='cdn')

# Check RACSEX
k=rac.melt(id_vars=['CT'],value_vars=['MALE','FEMALE'],var_name='PPSEX',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['CT','PPSEX'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPSEX'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='CT: '+k['CT']+'<br>'+'RACSEX: '+k['PPSEX']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'RACSEX (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/racsexpt.html',include_plotlyjs='cdn')

# Check RACAGE
k=rac.melt(id_vars=['CT'],value_vars=['AGE301','AGE302','AGE303'],var_name='RACAGE',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['CT','RACAGE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','RACAGE'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='CT: '+k['CT']+'<br>'+'RACAGE: '+k['RACAGE']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'RACAGE (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/racagept.html',include_plotlyjs='cdn')

# Check RACIND
k=rac.melt(id_vars=['CT'],value_vars=['AGR','EXT','UTL','CON','MFG','WHL','RET','TRN','INF','FIN',
                                      'RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM'],var_name='PPIND',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['CT','PPIND'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','PPIND'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='CT: '+k['CT']+'<br>'+'RACIND: '+k['PPIND']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'RACIND (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/racindpt.html',include_plotlyjs='cdn')



# Clean up WAC
wac=[]
for i in ['ct','nj','ny']:
    wac+=[pd.read_csv(path+'LEHD/'+i+'_wac_S000_JT01_2018.csv',dtype=float,converters={'w_geocode':str})]
wac=pd.concat(wac,axis=0,ignore_index=True)
wac.columns=['BK','TW','AGE301','AGE302','AGE303','EARN1','EARN2','EARN3','AGR','EXT','UTL','CON','MFG',
             'WHL','RET','TRN','INF','FIN','RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM',
             'WHT','BLK','NTV','ASN','PCF','TWO','NHS','HSP','LH','HS','SCAD','BDGD','MALE','FEMALE',
             'CFA01','CFA02','CFA03','CFA04','CFA05','CFS01','CFS02','CFS03','CFS04','CFS05','DATE']
wac=pd.merge(wac,geoxwalk,how='inner',left_on='BK',right_on='Block2010')
wac=wac[np.isin(wac['StateCounty'],bpm)].reset_index(drop=True)
wac=wac.groupby(['POWPUMA2010'],as_index=False).agg({'TW':'sum','AGE301':'sum','AGE302':'sum',
                                                     'AGE303':'sum','AGR':'sum','EXT':'sum',
                                                     'UTL':'sum','CON':'sum','MFG':'sum','WHL':'sum',
                                                     'RET':'sum','TRN':'sum','INF':'sum','FIN':'sum',
                                                     'RER':'sum','PRF':'sum','MNG':'sum','WMS':'sum',
                                                     'EDU':'sum','MED':'sum','ENT':'sum','ACC':'sum',
                                                     'SRV':'sum','ADM':'sum','LH':'sum','HS':'sum',
                                                     'SCAD':'sum','BDGD':'sum','MALE':'sum',
                                                     'FEMALE':'sum'}).reset_index(drop=True)
wac.columns=['POWPUMA','TW','AGE301','AGE302','AGE303','AGR','EXT','UTL','CON','MFG','WHL','RET',
             'TRN','INF','FIN','RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM','LH','HS',
             'SCAD','BDGD','MALE','FEMALE']
wac.to_csv(path+'LEHD/wac.csv',index=False)

# Validation
dfpp=pd.read_csv(path+'POP/dfpp.csv',dtype=str,converters={'PWGTP':float,'TOTAL':float})
dfppwk=dfpp[(dfpp['PPMODE']!='NW')&(dfpp['PPIND']!='MIL')].reset_index(drop=True)
dfppwk['WACAGE']=np.where(np.isin(dfppwk['PPAGE'],['AGE04','AGE05','AGE06']),'AGE301',
                 np.where(np.isin(dfppwk['PPAGE'],['AGE07','AGE08','AGE09','AGE10','AGE11']),'AGE302',
                 np.where(np.isin(dfppwk['PPAGE'],['AGE12','AGE13','AGE14','AGE15','AGE16','AGE17','AGE18']),'AGE303','OTH')))
wac=pd.read_csv(path+'LEHD/wac.csv',dtype=float,converters={'POWPUMA':str})
# WAC missing off-the-book workers
# Model missing residents outside BPM working in POWPUMAs

# Check WACTW
k=pd.merge(dfppwk.groupby(['POWPUMA'],as_index=False).agg({'TOTAL':'sum'}),wac[['POWPUMA','TW']],how='inner',on=['POWPUMA'])
k=k.sort_values('TW').reset_index(drop=True)
k['HOVER']='POWPUMA: '+k['POWPUMA']+'<br>'+'MODEL: '+k['TOTAL'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TW'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL']-k['TW'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL'],
                               y=k['TW'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL'])],
                               y=[0,max(k['TOTAL'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'WACTW (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/wactwpt.html',include_plotlyjs='cdn')

# Check WACSEX
k=wac.melt(id_vars=['POWPUMA'],value_vars=['MALE','FEMALE'],var_name='PPSEX',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['POWPUMA','PPSEX'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['POWPUMA','PPSEX'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='POWPUMA: '+k['POWPUMA']+'<br>'+'WACSEX: '+k['PPSEX']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'WACSEX (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/wacsexpt.html',include_plotlyjs='cdn')

# Check WACAGE
k=wac.melt(id_vars=['POWPUMA'],value_vars=['AGE301','AGE302','AGE303'],var_name='WACAGE',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['POWPUMA','WACAGE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['POWPUMA','WACAGE'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='POWPUMA: '+k['POWPUMA']+'<br>'+'WACAGE: '+k['WACAGE']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'WACAGE (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/wacagept.html',include_plotlyjs='cdn')

# Check WACIND
k=wac.melt(id_vars=['POWPUMA'],value_vars=['AGR','EXT','UTL','CON','MFG','WHL','RET','TRN','INF',
                                           'FIN','RER','PRF','MNG','WMS','EDU','MED','ENT','ACC',
                                           'SRV','ADM'],var_name='PPIND',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['POWPUMA','PPIND'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['POWPUMA','PPIND'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='POWPUMA: '+k['POWPUMA']+'<br>'+'WACIND: '+k['PPIND']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'WACIND (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/wacindpt.html',include_plotlyjs='cdn')



# Clean up ODCTPUMA
od=[]
for i in ['ct','nj','ny']:
    od+=[pd.read_csv(path+'LEHD/'+i+'_od_main_JT01_2018.csv',dtype=float,converters={'w_geocode':str,'h_geocode':str})]
    od+=[pd.read_csv(path+'LEHD/'+i+'_od_aux_JT01_2018.csv',dtype=float,converters={'w_geocode':str,'h_geocode':str})]
od=pd.concat(od,axis=0,ignore_index=True)
od.columns=['WBK','HBK','TW','AGE301','AGE302','AGE303','EARN1','EARN2','EARN3','IND1','IND2','IND3',
            'DATE']
od=pd.merge(od,geoxwalk,how='inner',left_on='HBK',right_on='Block2010')
od=od[np.isin(od['StateCounty'],bpm)].reset_index(drop=True)
od=pd.merge(od,geoxwalk,how='inner',left_on='WBK',right_on='Block2010')
od=od[np.isin(od['StateCounty_y'],bpm)].reset_index(drop=True)
od=od.groupby(['CensusTract2010_x','POWPUMA2010_y'],as_index=False).agg({'TW':'sum',
                                                                         'AGE301':'sum',
                                                                         'AGE302':'sum',
                                                                         'AGE303':'sum',
                                                                         'IND1':'sum',
                                                                         'IND2':'sum',
                                                                         'IND3':'sum'}).reset_index(drop=True)
od.columns=['CT','POWPUMA','TW','AGE301','AGE302','AGE303','IND1','IND2','IND3']
od.to_csv(path+'LEHD/odctpuma.csv',index=False)

# Validation
dfpp=pd.read_csv(path+'POP/dfpp.csv',dtype=str,converters={'PWGTP':float,'TOTAL':float})
dfppwk=dfpp[(dfpp['PPMODE']!='NW')&(dfpp['PPIND']!='MIL')].reset_index(drop=True)
dfppwk=pd.merge(dfppwk,geoxwalk[['POWPUMA2010','StateCounty']].drop_duplicates(keep='first'),how='inner',left_on='POWPUMA',right_on='POWPUMA2010')
dfppwk=dfppwk[np.isin(dfppwk['StateCounty'],bpm)].reset_index(drop=True)
dfppwk=dfppwk.drop(['StateCounty'],axis=1).reset_index(drop=True)
dfppwk=dfppwk.drop_duplicates(keep='first').reset_index(drop=True)
dfppwk['ODAGE']=np.where(np.isin(dfppwk['PPAGE'],['AGE04','AGE05','AGE06']),'AGE301',
                np.where(np.isin(dfppwk['PPAGE'],['AGE07','AGE08','AGE09','AGE10','AGE11']),'AGE302',
                np.where(np.isin(dfppwk['PPAGE'],['AGE12','AGE13','AGE14','AGE15','AGE16','AGE17','AGE18']),'AGE303','OTH')))
dfppwk['ODIND']=np.where(np.isin(dfppwk['PPIND'],['AGR','EXT','CON','MFG']),'IND1',
                np.where(np.isin(dfppwk['PPIND'],['UTL','WHL','RET','TRN']),'IND2',
                np.where(np.isin(dfppwk['PPIND'],['INF','FIN','RER','PRF','MNG','WMS','EDU','MED','ENT','ACC','SRV','ADM']),'IND3','OTH')))
od=pd.read_csv(path+'LEHD/odctpuma.csv',dtype=float,converters={'CT':str,'POWPUMA':str})
# OD missing off-the-book residents

# Check ODTW
k=pd.merge(dfppwk.groupby(['CT','POWPUMA'],as_index=False).agg({'TOTAL':'sum'}),od[['CT','POWPUMA','TW']],how='inner',on=['CT','POWPUMA'])
k=k.sort_values('TW').reset_index(drop=True)
k['HOVER']='CT： '+k['CT']+'<br>'+'POWPUMA: '+k['POWPUMA']+'<br>'+'MODEL: '+k['TOTAL'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TW'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL']-k['TW'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL'],
                               y=k['TW'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL'])],
                               y=[0,max(k['TOTAL'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'ODTW (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/odtwpt.html',include_plotlyjs='cdn')

# Check ODAGE
k=od.melt(id_vars=['CT','POWPUMA'],value_vars=['AGE301','AGE302','AGE303'],var_name='ODAGE',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['CT','POWPUMA','ODAGE'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','POWPUMA','ODAGE'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='CT: '+k['CT']+'<br>'+'POWPUMA: '+k['POWPUMA']+'<br>'+'ODAGE: '+k['ODAGE']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'ODAGE (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/odagept.html',include_plotlyjs='cdn')

# Check ODIND
k=od.melt(id_vars=['CT','POWPUMA'],value_vars=['IND1','IND2','IND3'],var_name='ODIND',value_name='TOTAL')
k=pd.merge(dfppwk.groupby(['CT','POWPUMA','ODIND'],as_index=False).agg({'TOTAL':'sum'}),k,how='inner',on=['CT','POWPUMA','ODIND'])
k=k.sort_values('TOTAL_x').reset_index(drop=True)
k['HOVER']='CT: '+k['CT']+'<br>'+'POWPUMA: '+k['POWPUMA']+'<br>'+'ODIND: '+k['ODIND']+'<br>'+'MODEL: '+k['TOTAL_x'].astype(int).astype(str)+'<br>'+'LEHD: '+k['TOTAL_y'].astype(int).astype(str)
rmse=np.sqrt(sum((k['TOTAL_x']-k['TOTAL_y'])**2)/len(k))
fig=go.Figure()
fig=fig.add_trace(go.Scattergl(name='LEHD vs MODEL',
                               x=k['TOTAL_x'],
                               y=k['TOTAL_y'],
                               mode='markers',
                               marker={'color':'rgba(44,127,184,1)',
                                       'size':5},
                               hoverinfo='text',
                               hovertext=k['HOVER']))
fig=fig.add_trace(go.Scattergl(name='OPTIMAL',
                               x=[0,max(k['TOTAL_x'])],
                               y=[0,max(k['TOTAL_x'])],
                               mode='lines',
                               line={'color':'rgba(215,25,28,1)',
                                     'width':2},
                               hoverinfo='skip'))
fig.update_layout(
    template='plotly_white',
    title={'text':'ODIND (RMSE: '+format(rmse,'.2f')+')',
       'font_size':20,
       'x':0.5,
       'xanchor':'center'},
    legend={'orientation':'h',
            'title_text':'',
            'font_size':16,
            'x':0.5,
            'xanchor':'center',
            'y':1,
            'yanchor':'bottom'},
    xaxis={'title':{'text':'MODEL',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    yaxis={'title':{'text':'LEHD',
                    'font_size':14},
           'tickfont_size':12,
           'rangemode':'nonnegative',
           'showgrid':True},
    font={'family':'Arial',
          'color':'black'},
)
fig.write_html(path+'POP/validation/odindpt.html',include_plotlyjs='cdn')



# Clean up ODCTCT
od=[]
for i in ['ct','nj','ny']:
    od+=[pd.read_csv(path+'LEHD/'+i+'_od_main_JT01_2018.csv',dtype=float,converters={'w_geocode':str,'h_geocode':str})]
    od+=[pd.read_csv(path+'LEHD/'+i+'_od_aux_JT01_2018.csv',dtype=float,converters={'w_geocode':str,'h_geocode':str})]
od=pd.concat(od,axis=0,ignore_index=True)
od.columns=['WBK','HBK','TW','AGE301','AGE302','AGE303','EARN1','EARN2','EARN3','IND1','IND2','IND3',
            'DATE']
od=pd.merge(od,geoxwalk,how='inner',left_on='HBK',right_on='Block2010')
od=od[np.isin(od['StateCounty'],bpm)].reset_index(drop=True)
od=pd.merge(od,geoxwalk,how='inner',left_on='WBK',right_on='Block2010')
od=od[np.isin(od['StateCounty_y'],bpm)].reset_index(drop=True)
od=od.groupby(['CensusTract2010_x','CensusTract2010_y'],as_index=False).agg({'TW':'sum',
                                                                             'AGE301':'sum',
                                                                             'AGE302':'sum',
                                                                             'AGE303':'sum',
                                                                             'IND1':'sum',
                                                                             'IND2':'sum',
                                                                             'IND3':'sum'}).reset_index(drop=True)
od.columns=['RACCT','WACCT','TW','AGE301','AGE302','AGE303','IND1','IND2','IND3']
od.to_csv(path+'LEHD/odctct.csv',index=False)







