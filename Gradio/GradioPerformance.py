

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import gradio as gr

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[1]:

drop_down_list=[]


def get_perf(ISIN, dateFrom, dateTo,type_code='ISINCodePtf'):

    url = f"https://perfo-api.intramundi.com/performance/api/v1/valuationData/flatten?typeCode={type_code}&assetcode={ISIN}&dateFrom={dateFrom}&dateTo={dateTo}"
    data = pd.read_json(url)
    data["valuationDate"] = pd.to_datetime(
        data["valuationDate"], errors="coerce", utc=True
    )
    data["valuationDate"] = data["valuationDate"].apply(
        lambda x: x.replace(tzinfo=None)
    )
    data = data.set_index("valuationDate")
    # data.index=data.index.normalize()

    return data

               
                

# In[ ]:


def load_excel(file):
    global ETF,ISINs
    
    try:
        ETF = pd.read_excel(file.name)
        ETF = ETF.set_index(ETF.columns[0])
        ISINs=ETF.index
        
        return "File uploaded successfully!"

    except Exception as e:
        return f"Error: {str(e)}"


# In[2]:


def get_return(dateFrom,dateTo,option):
    
    global perf_dict, not_found
    
    perf_dict = {}
    not_found = []
    for ISIN in ISINs:
        try:
            
            ISIN_Data = get_perf(ISIN, dateFrom, dateTo,type_code=option)

            if "benchPerf" in ISIN_Data.columns:

                ISIN_Data["Share Class Return"]=ISIN_Data["grossperf"]/100
                ISIN_Data["Benchmark Return"]=ISIN_Data["benchPerf"]/100
                ISIN_Data['Benchmark Return']=np.where(ISIN_Data['Benchmark Return']<=-1,0,ISIN_Data['Benchmark Return'])

                ISIN_Data["Excess Returns"] = (ISIN_Data["Share Class Return"] - ISIN_Data["Benchmark Return"])

            else:
                ISIN_Data["Share Class Return"]=ISIN_Data["grossperf"]/100
                ISIN_Data["Benchmark Return"]=ISIN_Data["Share Class Return"]*0

                ISIN_Data["Excess Returns"] = (ISIN_Data["Share Class Return"] - ISIN_Data["Benchmark Return"])

                
            perf_dict[ISIN] = ISIN_Data
            
            print(ISIN)
            
        except Exception as e:

            print(f"Data not found for {ISIN}")
            not_found.append(ISIN)
            pass
        
    return "Computation Done!" ,pd.DataFrame(not_found,columns=["Not Found"]),pd.DataFrame(perf_dict.keys(),columns=['Scope'])


# In[10]:


def atypical_control(basis_point_limit_daily=0.01,basis_point_limit_cumulative=0.03):
    
    # Global Variables to be used in every Gradio Function
    global flagged_cumulative,flagged_daily
    
    
    flagged_cumulative = {}
    flagged_daily = {}
    
    #Daily Control and Cumulative Control on Excess Returns to flag atypical Performance#

    te={}
    vol={}
    for key in perf_dict:
        temp=(perf_dict[key]['Excess Returns']).resample('ME').std()*np.sqrt(252)
        temp_vol=(perf_dict[key]['Share Class Return']).resample('ME').std()*np.sqrt(252)
        te[key]=temp
        vol[key]=temp_vol


    for ISIN in perf_dict:

        temp = perf_dict[ISIN]
        index = np.where(abs(temp["Excess Returns"]) > basis_point_limit_daily/10000)
        flagged_daily[ISIN] = temp.iloc[index]

        if (
            abs((1 + temp["Excess Returns"]).cumprod().iloc[-1] - 1)
            > basis_point_limit_cumulative/10000
        ):
            flagged_cumulative[ISIN] = (1 + temp["Excess Returns"]).cumprod()

        else:

            continue
    
    # Compute the cumulative Excess Returns to be used as a plot
    excess_returns_cumulative={}
    for key in perf_dict:
        excess_returns_cumulative[key]=(1+perf_dict[key]["Excess Returns"]).cumprod()
    
    #excess_returns_cumulative_dataframe=pd.DataFrame(excess_returns_cumulative)
    
    #Get last value of Cumulative Excess Return (to see if above limit and be inserted in a table)
    
    cumulative = {}
    for ISIN in perf_dict:
        temp = perf_dict[ISIN]
        cumulative[ISIN] = ((1 + temp["Excess Returns"]).cumprod().iloc[-1] - 1)*10000
    
    #Summary Table for the daily returns and ETFs flagged#
    
    summary_daily = {}

    for key in flagged_daily:

        try:
            temp = flagged_daily[key]
            count = temp.shape[0]
            max_dev = (temp["Excess Returns"].max()*10000).round(4)
            min_dev = (temp["Excess Returns"].min()*10000).round(4)
            date_max = temp["Excess Returns"].idxmax()
            date_min = temp["Excess Returns"].idxmin()

            summary_daily[key] = [count, max_dev, date_max, min_dev, date_min]

        except Exception as e:
            print(f"Data not found for {key}")

            pass
    
    excess = {}
    benchmark = {}
    share_class_returns = {}

    for key in perf_dict:
        
        temp=perf_dict[key]
        wo_dup=temp[~temp.index.duplicated()]
        excess[key] = wo_dup["Excess Returns"] 

        benchmark[key] = wo_dup["Benchmark Return"]
            
        share_class_returns[key] = wo_dup["Share Class Return"]
        
    
    global daily_deviation
    
    daily_deviation = pd.DataFrame(
    summary_daily,
    index=[
        "Numbers of Violations",
        "Max Upside Deviation in BPS",
        "Date of Max (Upside) Deviation",
        "Max Downside Deviation in BPS",
        "Date of Max (Downside) Deviation"]
    ).T
    
    global returns_dataframe,benchmark_returns,excess_returns_dataframe,cumulative_dataframe,monthly_te,monthly_vol
    
    returns_dataframe = pd.DataFrame(share_class_returns).sort_index()
    benchmark_returns = pd.DataFrame(benchmark).sort_index()
    excess_returns_dataframe = pd.DataFrame(excess).sort_index()

    monthly_te=pd.DataFrame(te).sort_index()
    monthly_vol=pd.DataFrame(vol).sort_index()

    cumulative_dataframe = pd.DataFrame(
        cumulative.values(), index=cumulative.keys(), columns=["Final Excess Return (Bps)"]
    )
    cumulative_dataframe = cumulative_dataframe.sort_values(
        by="Final Excess Return (Bps)", ascending=False
    )

    fund_list = list(perf_dict.keys())

    return daily_deviation.reset_index().round(4),cumulative_dataframe.loc[flagged_cumulative.keys()].reset_index().round(4),monthly_te.reset_index().round(6),gr.update(choices=fund_list, value=None)

def build_excel():

    with pd.ExcelWriter("Atypical Performance.xlsx", engine="openpyxl") as writer:

        returns_dataframe.to_excel(writer, sheet_name="Returns", index=True)
        benchmark_returns.to_excel(writer, sheet_name="Benchmark", index=True)
        excess_returns_dataframe.to_excel(writer, sheet_name="Excess Returns", index=True)
        daily_deviation.to_excel(writer, sheet_name="Daily Violations", index=True)
        cumulative_dataframe.to_excel(
            writer, sheet_name="Cumulative Violations", index=True)
        monthly_te.to_excel(writer, sheet_name="Monthly Tracking Error in %", index=True)
        monthly_vol.to_excel(writer, sheet_name="Monthly Volatility in %", index=True)

    return "Excel Downloaded"

def get_time_series(value1):

    fund=returns_dataframe[value1]
    bench=benchmark_returns[value1]
    excess=excess_returns_dataframe[value1]

    returns_series=pd.concat([fund,bench,excess],axis=1)
    returns_series.columns=['Fund','Benchmark','Excess Return']

    returns_series=(1+returns_series).cumprod()*100

    return returns_series


def get_monthly_tracking_error(value1):

    excess=excess_returns_dataframe[value1]
    excess.columns=['Monthly Tracking Error in %']
    monthly_tracking_error=excess.resample('ME').std()*np.sqrt(252)*100

    return monthly_tracking_error

def get_monthly_vol(value1):

    returns=returns_dataframe[value1].dropna()
    returns.columns=['Monthly Tracking Error in %']
    monthly_volatility=returns.resample('ME').std()*np.sqrt(252)*100

    return monthly_volatility

def plot_chart(value1):

    time_series=get_time_series(value1).dropna()
    monthly_te=get_monthly_tracking_error(value1).dropna()
    monthly_vol=get_monthly_vol(value1).dropna()

    fig = px.line(time_series[['Fund','Benchmark']], title="Fund",color_discrete_sequence = px.colors.sequential.Sunsetdark,render_mode='svg')
    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white") 
    fig.update_traces(textfont=dict(family="Arial Narrow"))

    fig2 = px.line(time_series['Excess Return'], title="Excess Return",color_discrete_sequence = px.colors.sequential.Sunsetdark,render_mode='svg')
    fig2.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white") 
    fig2.update_traces(textfont=dict(family="Arial Narrow"))

    fig3 = px.line(monthly_te, title="Monthly Tracking Error in %",color_discrete_sequence = px.colors.sequential.Sunsetdark,render_mode='svg')
    fig3.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white") 
    fig3.update_traces(textfont=dict(family="Arial Narrow"))


    fig4 = px.line(monthly_vol, title="Monthly Volatility in %",color_discrete_sequence = px.colors.sequential.Sunsetdark,render_mode='svg')
    fig4.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white") 
    fig4.update_traces(textfont=dict(family="Arial Narrow"))

    return fig,fig2,fig3,fig4,time_series.reset_index()


# In[ ]:


with gr.Blocks(css="* { font-family: 'Arial Narrow', sans-serif; }") as app:    

    with gr.Tab("Performance Control"):
        gr.Markdown("## Performance Control")
    
        # Define UI elements first
        file_input = gr.File(label="Upload Excel File (.xlsx)")
        file_status = gr.Textbox(label="Status", interactive=False)
        file_input.change(load_excel, inputs=file_input, outputs=[file_status])
        option_selected = gr.Dropdown(choices=['DecalogCodePtf','ISINCodePtf'],value='ISINCodePtf', label="Options")

        found = gr.Dataframe(label="Found ISINs", interactive=False)
        not_found = gr.Dataframe(label="Not Found ISINs", interactive=False)

        start_date=gr.Textbox(label="Start Date (YYYYMMDD)")
        end_date=gr.Textbox(label="End Date (YYYYMMDD)")
        return_status=gr.Textbox(label="Computation Status")
        
        get_return_button=gr.Button("Get Returns")
        get_return_button.click(fn=get_return, inputs=[start_date,end_date,option_selected], outputs=[return_status,not_found,found])
        
        daily_limit = gr.Number(label="Daily Limit (Bps)")
        cumulative_limit = gr.Number(label="Cumulative Limit (Bps)")

        daily_table = gr.Dataframe(label="Daily Deviation", interactive=False)
        cumulative_table=gr.Dataframe(label="Cumulative Deviation", interactive=False)
        tracking_error_table=gr.Dataframe(label="Tracking Error Monthly", interactive=False)


        atypical_check_button=gr.Button("Get Atypical Performance")
        excel_button=gr.Button("Get Excel Summary")

        
    with gr.Tab("Time Series"):
        

        # Define the dropdown only in the Time Series tab
        dropdown = gr.Dropdown(choices=[], label="ISIN")  # Initially empty

        time_series_plot = gr.Plot(label="Performance")
        excess_series_plot=gr.Plot(label="Excess Return")

        monthly_te_plot=gr.Plot(label="Monthly Tracking Error in %")
        monthly_vol_plot=gr.Plot(label="Monthly Volatility in %")

        fund_table = gr.Dataframe(label="Returns", interactive=False)

        dropdown.change(fn=plot_chart, inputs=dropdown, outputs=[time_series_plot,excess_series_plot,monthly_te_plot,monthly_vol_plot,fund_table])


# Connect the update function to the button click
    atypical_check_button.click(fn=atypical_control, inputs=[daily_limit,cumulative_limit], outputs=[daily_table,cumulative_table,tracking_error_table,dropdown])
    excel_file_status = gr.Textbox(label="Status", interactive=False)

    excel_button.click(fn=build_excel,inputs=[],outputs=[excel_file_status])

        
port=7860
app.launch(debug=True, share=False, server_port=port, server_name='0.0.0.0', root_path=f'/studio/vscode/{os.getenv("ALTO_STUDIO_USERNAME")}/proxy/{port}/')        
                             
        

