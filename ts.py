import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
st.set_option('deprecation.showPyplotGlobalUse', False)
from statsmodels.tsa.arima_model import ARIMA
def err(x,y):
        bias = x-y
        bias = bias.mean()
        mae = np.absolute(x-y)
        mae = mae.mean()
        mape = ((x-y)/x)*100
        mape = mape.mean()
        mse = (x-y)**2
        mse = mse.mean()
        a = [bias,mae,mape,mse]
        return a
def adfuller_test(data):
        result=adfuller(data)
        labels = ['ADF Test Statistic','p-value','Number of Lags Used','Number of Observations Used']
        for value,label in zip(result,labels):
            st.write(label+' : '+str(value))
        if result[1] <= 0.05:
            st.write("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            st.write("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
def decompose_series(ts):
    '''
    This function applies a seasonal decomposition to a time series. It will generate a season plot, a trending plot, and, finally, a resid plot

    Args.
        ts (Pandas Series): a time series to be decomposed
    '''
    fig = plt.Figure(figsize=(12,7))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    try:
        decomposition = seasonal_decompose(ts.asfreq('MS'))

    except AttributeError:
        error_message = '''
                        Seems that your DATE column is not in a proper format. 
                        Be sure that it\'s in a valid format for a Pandas to_datetime function.
                        '''
        raise AttributeError(error_message)

    decomposition.seasonal.plot(color='green', ax=ax1, title='Seasonality')
    plt.legend('')
    #plt.title('Seasonality')
    #st.pyplot()

    decomposition.trend.plot(color='green', ax=ax2, title='Trending')
    plt.legend('')
    #plt.title('Trending')
    #st.pyplot()
    
    decomposition.resid.plot(color='green', ax=ax3, title='Resid')
    plt.legend('')
    #plt.title('Resid')
    plt.subplots_adjust(hspace=1)
    st.pyplot() 
st.markdown("<h1 style='font-size:50px;text-align: center; color: black;'>AUTS</h1>", unsafe_allow_html=True)
lt,rt = st.sidebar.beta_columns(2)
pi = lt.selectbox('Select starting Auto-Regeressive order', [i for i in range(1,100)])
pe = rt.selectbox('Select ending Auto-Regeressive order', [i for i in range(1,100)])
lt1,rt1 = st.sidebar.beta_columns(2)
di = lt1.selectbox('Select starting Differencing order', [i for i in range(0,3)])
de = rt1.selectbox('Select ending Differencing order', [i for i in range(0,3)])
lt2,rt2 = st.sidebar.beta_columns(2)
qi = lt2.selectbox('Select starting Moving Average order', [i for i in range(0,100)])
qe = rt2.selectbox('Select ending Moving Average order', [i for i in range(0,100)])
uploaded_files = st.sidebar.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
st.sidebar.write('           ')
st.sidebar.write('           ')
st.sidebar.write('           ')
st.sidebar.write('           ')
st.sidebar.write('           ')

if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
    if str(file.name).endswith('v'):
        uploaded_data_read = [pd.read_csv(file) for file in uploaded_files]
    else:
        uploaded_data_read = [pd.read_excel(file) for file in uploaded_files]
    df = pd.concat(uploaded_data_read)
    st.write(df)
    df = df[df.iloc[:,1].notna()]
    st.write('File Chosen')
    st.write('Missing valus are eliminated')
    df.columns = ['time','series']
    df['series'] = df['series'].astype('float')
    st.write('Columns Aligned')
    df.reset_index(inplace = True) 
    st.write('Index reset')
    df.index = pd.to_datetime(df['time'])
    df.drop(columns = ['index','time'],inplace = True)
    st.write('Unnecessary Columns are removed')
    st.write(df)
    decompose_series(df)
    st.write('Successfully Generated Decomposition plot.')
    from statsmodels.tsa.stattools import adfuller
    adfuller_test(df)
    st.write(10)
    st.write(1)
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(df['series'].iloc[12:],ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(df['series'].iloc[12:],lags=40,ax=ax2)
    st.pyplot()
    lst = []
    mods = []
    bstmd = 0
    bstord = []
    for i in range(int(pi),int(pe)+1):
      for j in range(int(di),int(de)+1):
        for k in range(int(qi),int(qe)+1):
          ar = ARIMA(endog=df['series'],order=(i,j,k)).fit(disp=False)
          mods.append(ar)
          st.write(i,j,k)
          if bstmd==0:
            bstmd = ar
            bstord = (i,j,k)
            erl = err(df['series'],ar.predict())
          else:
            cnt = 0
            erl1 = err(df['series'],ar.predict())
            for z in range(0,len(erl1)):
              if erl1[z] < erl[z]:
                cnt += 1
            if cnt>3:
              bstmd = ar
              bstord = (i,j,k)
    st.write('Best Order is found......')
    st.write(bstord)
    st.write('Plotting actual values and predicted values......')
    df.plot(figsize=(25,8))
    bstmd.predict().plot(figsize=(25,8))
    st.pyplot()
    ind6 = abs(int(len(df)/6))
    ind46 = abs(int(len(df)*4/6))
    df['prd101'] = None
    df['act'] = None
    df['prd101'].iloc[ind6:ind46] = bstmd.predict().iloc[ind6:ind46]
    df['act'].iloc[:ind6] = df['series'].iloc[:ind6]
    df['act'].iloc[ind46:] = df['series'].iloc[ind46:]
    st.write('Plotting A concatenated plot of Actual and Predicted values........')
    df[['act','prd101']].plot(figsize=(25,8))
    st.pyplot()
    strt = st.text_input('Enter the initial month to predict.......')
    end = st.text_input('Enter the final month to predict.......')
    btns = st.button('Predict')
    if strt and btns and end:
        st.write(bstmd.predict(int(strt)-1,int(end)-1))
