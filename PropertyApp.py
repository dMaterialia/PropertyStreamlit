import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import copy
from sklearn.ensemble import RandomForestRegressor

st.title('Swansea Property Price Predictor')

@st.cache(allow_output_mutation=True)
def load_data():
    """
    Loads the data
    """
    import gdown
    url = 'https://drive.google.com/uc?id=1OD2l7ynVzLlqY92gYiCx5xq32-D1wJMe'
    output = 'one1.pkl'
    gdown.download(url, output)

    cda = os.getcwd()
    with open('one1.pkl', 'rb')as f: 
        m2, to2, xsAll2, yAll2 = pickle.load(f)
    return m2, to2, xsAll2, yAll2
data_load_state = st.text('Loading data...')
m1, to1, xsAll1, yAll1=load_data()
data_load_state.text("Loaded data (using st.cache)")

def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    import re
    import pandas as pd
    import numpy as np
    
    def ifnone(a, b):
        "`b` if `a` is None else `a`"
        return b if a is None else a
    
    def make_date(df, date_field):
        "Make sure `df[date_field]` is of the right date type."
        
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
    
    
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask,field.values.astype(np.int64) // 10 ** 9,np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

def get_predTodayNotExact(m,address,toTEMP,xs_final,y):
    """
    Given model m, address, initial pd of houses to, and adjusted pd xs_final
    output is house price prediction
    """
    import copy
    # convert to current date
    colsAll=xs_final.columns
    colsNoDate=['Type', 'Index of Multiple Deprivation', 'Latitude',
                'Average Income', 'Longitude', 'Postcode', 'Introduced', 'Address',
           'Altitude']
    xsNoDate=copy.copy(xs_final.drop(columns=['Elapsed','Year']))
    
    xsNoDate['Date'] = pd.to_datetime("today")
    xsNoDate = add_datepart(xsNoDate, 'Date')
    xs_finalTEMP=xsNoDate.loc[:,colsAll]

    # each address has a unique number
    aa=toTEMP.classes['Address']
    # findwhich number is address give (take 1st if more than 1)
    try:
        ii=[ii for ii,aa1 in enumerate(aa) if aa1== address][0]
        # 1 address can have multiple sales so we need index in dataframes
        ii=toTEMP[toTEMP['Address']==ii].index[0]

        preda = np.round( np.exp( m.predict(xs_finalTEMP.loc[ii:ii]) )/1000 ,1)
        prev = np.round( np.exp(y.loc[ii])/1000 ,1)
        
        typeAll=toTEMP.classes['Type']
        typa=typeAll[xs_finalTEMP.loc[ii:ii,'Type']][0]
    
    except:
        aa=toTEMP.classes['Address']
        aaStreet=toTEMP.classes['Street']
        ii=[ii for ii,aa1 in enumerate(aaStreet) if aa1== Street][0]
        xsTemp=copy.copy( xs_finalTEMP[xs_finalTEMP['Street']==ii] )
        xsTemp.reset_index(inplace=True,drop=True)
        # find nearest house by houseno
        No=np.array(xsTemp['HouseNo'])

        yo=(np.abs(No-HouseNo))
        yo1=np.min(yo)
        # get index of the nearest house
        yo=No[yo==yo1][0]  

        ii=[ii for ii,aa1 in enumerate(xsTemp.HouseNo) if aa1== yo][0]
        xsTemp.loc[ii:ii,'HouseNo']=HouseNo
        # If want to change house type
        
#         xsTemp.loc[ii:ii,'Type']=2
#         print(xsTemp.loc[ii:ii,'Type'])
#         print(xsTemp.loc[ii:ii])
        
        preda = np.round( np.exp( m.predict(xsTemp.loc[ii:ii]) )/1000 ,1)[0]
        prev=0
        
        typeAll=toTEMP.classes['Type']
        typa=typeAll[xsTemp.loc[ii:ii,'Type']][0]
        
        
    return preda, prev, typa


def doSelect(typee,option2,typeeOut,toTEMP):
        streetAll=toTEMP.classes[typee]
        AdAll=toTEMP.classes[typeeOut]
        # this finds index of postcode for example SA1 0EA = 62
        indexPC1=[ita for ita,ij in enumerate(streetAll) if ij==option2][0]

        # finds all indexes of addresses with given post code index 
        indexAdds=[ita for ita, ij in enumerate(toTEMP[typee]) if ij==indexPC1]

        # Find address index numbers for those given above
        indexAddSel=toTEMP.iloc[indexAdds][typeeOut]

        # Convert these to actual addresses
        AdSel=AdAll[indexAddSel]
        
        # unique values
        AdSel=np.unique(AdSel)
        return AdSel
    

pcodesSA=['SA1', 'SA2', 'SA3', 'SA4', 'SA5', 'SA6', 'SA7', 'SA8',          
          'SA9', 'SA10' ,'SA11', 'SA12', 'SA13', 'SA14','SA15','SA18']

choice=['Post Code','Region', 'Street']

to=copy.copy(to1)
m=copy.copy(m1)
xsAll=copy.copy(xsAll1)
yAll=copy.copy(yAll1)
# These are the list of all addresses etc by actual name
AdAll=(to.classes['Address'])
pcAll=(to.classes['Postcode'])
regionAll=(to.classes['Region'])
streetAll=(to.classes['Street'])

# An optionbox- Select How search
optionSELECT = st.sidebar.selectbox(
    'Select how to search',
     choice)

if optionSELECT=='Post Code':
    # An optionbox- Select Postcode Start e.g. SA1
    option = st.sidebar.selectbox(
        'Select Area',
         pcodesSA)


    # Select Postcode All

    # This finds a set of postcodes given by optionbox
    indexPCSA=[ij for ij in pcAll if ij.split(' ')[0]==option]

    # optionbox to select particular postcode 
    # Outcome e.g. SA1 0EA
    option2 = st.sidebar.selectbox(
        'Select Postcode',
        indexPCSA)

    AdSel = doSelect(typee='Postcode',option2=option2,typeeOut='Address',toTEMP=(to))
    

elif optionSELECT=='Region':

    
    option2 = st.sidebar.selectbox(
         'Select Region',
         regionAll)
    
    StreetSel = doSelect(typee='Region',option2=option2,typeeOut='Street',toTEMP=(to))
       
        
    option3 = st.sidebar.selectbox(
         'Select Street',
         StreetSel)
    
    AdSel = doSelect(typee='Street',option2=option3,typeeOut='Address',toTEMP=(to))
        
    

elif optionSELECT=='Street':

    
    option2 = st.sidebar.selectbox(
         'Select Street',
         streetAll)
    
    AdSel = doSelect(typee='Street',option2=option2,typeeOut='Address',toTEMP=(to))
    
    

address = st.sidebar.selectbox(
    'Select Address',
    AdSel)

Pri1, Pri2, typa=get_predTodayNotExact(m,address,(to),(xsAll),(yAll))

#tell user what they selected
'You selected: ', option2, 'and', address
'Property type is ',typa

stra = 'The predicted price is: '
st.subheader(stra)
st.header('£'+ str(Pri1[0])+'k')



