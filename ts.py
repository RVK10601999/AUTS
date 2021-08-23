import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
st.set_option('deprecation.showPyplotGlobalUse', False)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import base64,pickle
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
adfvl = 0
from statsmodels.tsa.stattools import adfuller
def adfuller_test(data):
        global adfvl
        result=adfuller(data['series'])
        labels = ['ADF Test Statistic','p-value','Number of Lags Used','Number of Observations Used']
        for value,label in zip(result,labels):
            st.write(label+' : '+str(value))
        if result[1] <= 0.05:
            st.write("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            st.write("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
            adfvl = 1
def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href
def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;some_name&gt;.pkl)'
    return href
from statsmodels.tsa.stattools import kpss
kpsvl = 0
def kpss_test(series, **kw):
    global adfvl
    statistic, p_value, n_lags, critical_values = kpss(series['series'].notna(), **kw)
    # Format Output
    st.write(f'KPSS Statistic: {statistic}')
    st.write(f'p-value: {p_value}')
    st.write(f'num lags: {n_lags}')
    st.write('Critial Values:')
    for key, value in critical_values.items():
        st.write(f'   {key} : {value}')
    if p_value<.05:
        st.write('Not stationary')
        kpsvl = 1
    else:
        st.write('stationary')
    return adfvl
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
uploaded_files = st.sidebar.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
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
    st.write('Missing values are eliminated')
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
    adfuller_test(df)
    kpss_test(df)
    #st in both cases
    if 'df' not in st.session_state:
        st.session_state.df = df
    if 'cntr' not in st.session_state:
        st.session_state.cntr = 0
    lg_trns = st.radio('Please, select an option.....',options = ('options','log transform','differencing'))
    apl = st.button('apply..')
    if lg_trns == 'log transform' and apl:
        st.session_state.cntr+=1
        st.session_state.df[f'orig_log_{st.session_state.cntr}'] = st.session_state.df['series']
        st.session_state.df['series'] = np.log(st.session_state.df['series'])
        
    elif lg_trns == 'differencing':
        gap = st.text_input('enter gap')
        if gap:
            apl1 = st.button('apply...')
            if apl1:
                st.session_state.cntr+=1
                st.session_state.df[f'orig_diff_{st.session_state.cntr}'] = st.session_state.df['series']
                st.session_state.df['series'] = st.session_state.df['series'] - st.session_state.df['series'].shift(5)
    st.write(st.session_state.df.head(5))
    df = st.session_state.df
    arima_build = st.button('BUILD ARIMA MODEL')
    srmx_build = st.button('BUILD SARIMAX MODEL')
    auto_arima_b = st.button('BUILD ARIMA MODEL with auto_arima')
    auto_srmx_b = st.button('BUILD SARIMAX MODEL with auto_arima')
    tst = st.button('test again for stationarity...')
    strt = st.text_input('Enter the initial month to predict.......')
    end = st.text_input('Enter the final month to predict.......')
    
    if tst:
        adfuller_test(df)
        kpss_test(df)
    if arima_build:
        st.write(0,0)
        st.write(11)
        st.write(2)
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
        for i in range(0,5):
            for j in range(0,2):
                for k in range(0,5):
                    try:
                        ar = ARIMA(endog=df['series'],order=(i,j,k)).fit(disp=False)
                        mods.append(ar)
                        
                        if bstmd==0:
                            bstmd = ar
                            bstord = (i,j,k)
                            bstlord = 'SARIMAX not used'
                            erl = err(df['series'],ar.predict())
                            st.write(i,j,k)
                        else: 
                            cnt = 0
                            erl1 = err(df['series'],ar.predict())
                            for z in range(0,len(erl1)):
                                if erl1[z] < erl[z]:
                                    cnt += 1
                            if cnt>=3:
                                bstmd = ar
                                bstord = (i,j,k)
                                st.write(i,j,k)
                                bstlord = 'SARIMAX not used'
                    except: 
                        continue
                        st.write('continuing')
            #st in adf and non-st in kpss                    
            #non-st in both cases        
        st.write('Best Order is found......')
        st.write(bstord)
        st.write(bstlord)
        st.write(bstmd)
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
        resdf = bstmd.predict(int(strt)-1,int(end)-1)
        st.write(resdf.head())
        st.markdown(get_table_download_link_csv(resdf), unsafe_allow_html=True)
        st.markdown(download_model(bstmd), unsafe_allow_html=True)

    elif srmx_build:
        
        st.write(1,0)
        st.write(11)
        st.write(2)
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
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    for i1 in range(0,3):
                        for j1 in range(0,3):
                            for k1 in range(0,3):
                                try:
                                    ar = SARIMAX(endog=df['series'],order=(i,j,k),seasonal_order=(i1,j1,k1,12)).fit(disp=False)
                                    mods.append(ar)
                                    st.write(i,j,k,(i1,j1,k1))
                                    if bstmd==0:
                                        bstmd = ar
                                        bstord = (i,j,k)
                                        bstlord = (i1,j1,k1,12)
                                        erl = err(df['series'],ar.predict())
                                    else: 
                                        cnt = 0
                                        erl1 = err(df['series'],ar.predict())
                                        for z in range(0,len(erl1)):
                                            if erl1[z] < erl[z]:
                                                cnt += 1
                                        if cnt>=3:
                                            bstmd = ar
                                            bstord = (i,j,k)
                                            bstlord = (i1,j1,k1,12)
                                except: 
                                    continue
                                    st.write('continuing')
        st.write('Best Order is found......')
        st.write(bstord)
        st.write(bstlord)
        st.write(bstmd)
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
        resdf = bstmd.predict(int(strt)-1,int(end)-1)
        st.write(resdf.head())
        st.markdown(get_table_download_link_csv(resdf), unsafe_allow_html=True)
        st.markdown(download_model(bstmd), unsafe_allow_html=True)

        st.write('Best Order is found......')
        st.write(bstord)
        st.write(bstlord)
        st.write(bstmd)
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
        resdf = bstmd.predict(int(strt)-1,int(end)-1)
        st.write(resdf.head())
        st.markdown(get_table_download_link_csv(resdf), unsafe_allow_html=True)
        st.markdown(download_model(bstmd), unsafe_allow_html=True)        
                           

    
    
    