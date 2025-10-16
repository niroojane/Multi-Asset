import datetime
import json
import os
import time
from multiprocessing import Pool, cpu_count

import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from IPython.display import HTML, Markdown, display
from requests.auth import HTTPBasicAuth
from smart_services import SmartServices
from suds.client import Client
from Add_In_Queries import *

def display_scrollable_df(df, max_height="50vh", max_width="90vw"):
    style = f"""
    <div style="
        display: flex;
        justify-content: center;
        padding: 20px;
    ">
        <div style="
            overflow: auto;
            max-height: {max_height};
            max-width: {max_width};
            width: 100%;
            border: 1px solid #444;
            padding: 10px;
            background-color: #000;
            color: #eee;
            font-family: 'Arial Narrow', Arial, sans-serif;
            box-sizing: border-box;
        ">
            {df.to_html(classes='table', border=0, index=True)}
        </div>
    </div>
    """
    return HTML(style)

def get_pra(date,alto_smartservices):

    results = alto_smartservices.get(
        query_runner="PRA_all_Fields_scopes",
        params={"format": "json", "PRA_Start_Date": date, "PRA_end_historical": date},
    )

    return results

def safe_call_fields(tasks):
    
    func,fund,field,dateTo=tasks
    
    try:
        return dateTo, field,fund,func(fund,field,dateTo,dateTo)
    except Exception as e:
        return {dateTo: str(e), "args": tasks}
    

def safe_call_pra(tasks):
    func, date,sessions = tasks

    try:
        return date, func(date,sessions)
    except Exception as e:
        return {date: str(e), "args": tasks}
    
    
def display_ex_ante_app(sessions):

    
    funds=[]
    fields=[]

    date_start = widgets.Text(
        step=1,
        description="Start (YYYY-MM-DD)",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "auto"},
    )

    end_date = widgets.Text(
        step=1,
        description="End (YYYY-MM-DD)",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "auto"},
    )

    def get_data_on_click(b):

        global tables, fields,funds

        start = date_start.value
        end = end_date.value

        date_list = pd.date_range(start, end, freq="W-FRI").strftime("%Y-%m-%d")

        # date_list=pd.bdate_range(start, end, freq='W-MON').strftime('%Y-%m-%d')
        tasks = [(get_pra, date,sessions) for date in date_list]

        dico_data = {}
        start = time.time()
        with Pool(processes=min(5, cpu_count() * 2)) as pool:
            for date, data in pool.imap_unordered(safe_call_pra, tasks):
                if type(data) is not str:
                    dico_data[date] = data
                    print(f'data found at {date}')
                else:
                    print(date)

        dataframe = pd.DataFrame()
        for date in dico_data:
            dataframe = pd.concat([dataframe, dico_data[date]])

        tables = {}
        AIL = dataframe[dataframe["COMP_LABEL"] == "AMUNDI.IRL"].copy()
        AIL["REPORT_DATE"] = pd.to_datetime(
            AIL["REPORT_DATE"], errors="coerce", utc=True
        )
        AIL["REPORT_DATE"] = AIL["REPORT_DATE"].apply(lambda x: x.replace(tzinfo=None))
        list_of_funds = set(AIL["SELECTEDFUNDCODE"])
        tables["Scope"] = pd.DataFrame(list_of_funds, columns=["Scope"])
        tables["Data"] = AIL
        tables["Fields"] = pd.DataFrame(AIL.columns, columns=["Ex Ante Fields"])

        funds=tables["Scope"]["Scope"]
        fields=tables["Fields"]["Ex Ante Fields"]

        dropdown1.options = fields
        dropdown2.options = funds

        def get_excel(b):

            with pd.ExcelWriter("Ex Ante Data.xlsx", engine="openpyxl") as writer:

                tables["Data"].to_excel(writer, sheet_name="Data", index=True)
                tables["Scope"].to_excel(writer, sheet_name="Scope", index=True)
                tables["Fields"].to_excel(writer, sheet_name="Scope", index=True)

            print("File Generated")

        bt_excel = widgets.Button(
            description="Get Excel",
            layout=widgets.Layout(
                display="flex",
                justify_content="center",
                align_items="center",
                spacing="10px",
                width="auto",
            ),
        )

        bt_excel.on_click(get_excel)
        with data_output:
            data_output.clear_output()
            display(display_scrollable_df(tables["Scope"]))
            display(display_scrollable_df(tables["Fields"]))
            # display(display_scrollable_df(get_perf_catalog()))
            display(bt_excel)
            # display(display_scrollable_df(tables['Data']))


    dropdown1 = widgets.Dropdown(description="Fields:", value=None, options=fields)
    dropdown2 = widgets.Dropdown(description="Funds:", value=None, options=funds)

    data_output = widgets.Output()
    button_data = widgets.Button(description="Get Data")
    button_data.on_click(get_data_on_click)

    parameters_ui = widgets.VBox(
        [
            widgets.HBox(
                [date_start, end_date, button_data],
                layout=widgets.Layout(
                    display="flex",
                    justify_content="center",
                    align_items="center",
                    spacing="auto",
                    width="auto",
                ),
            ),
            data_output,
        ]
    )

    data = []


    def on_add_constraint_clicked(b):
        row = {"Field": dropdown1.value, "Fund": dropdown2.value}
        data.append(row)
        with constraint_output:
            constraint_output.clear_output()
            display(pd.DataFrame(data))


    add_constraint_btn = widgets.Button(description="Add Filter", button_style="success")
    add_constraint_btn.on_click(on_add_constraint_clicked)

    constraint_output = widgets.Output()
    output = widgets.Output()


    def on_clear_clicked(b):
        data.clear()
        res.clear()
        with constraint_output:
            constraint_output.clear_output()
            display(pd.DataFrame(columns=["Field", "Fund"]))

        with output:
            output.clear_output()


    clear_btn = widgets.Button(description="Clear All", button_style="danger")
    clear_btn.on_click(on_clear_clicked)

    res = {}


    def on_optimize_clicked(b):

        filter_dataframe = pd.DataFrame(data)
        unique_list_funds = set(filter_dataframe["Fund"])
        dico_filter = {}
        for fund in unique_list_funds:
            temp = filter_dataframe[filter_dataframe["Fund"] == fund]
            dico_filter[fund] = list(set(temp["Field"]))

        for key in dico_filter:

            temp = tables["Data"][tables["Data"]["SELECTEDFUNDCODE"] == key].set_index(
                "REPORT_DATE"
            )[dico_filter[key]]
            res[key] = temp

        with output:
            output.clear_output()
            for key in res:
                display(Markdown("### " + str(key)))
                display(display_scrollable_df(res[key]))


    optimize_btn = widgets.Button(description="Filter", button_style="primary")
    optimize_btn.on_click(on_optimize_clicked)

    constraint_ui = widgets.VBox(
    [
        widgets.VBox([dropdown1, dropdown2]),
        widgets.HBox([add_constraint_btn, clear_btn, optimize_btn]),
        constraint_output,
        output,
    ]
        )

    tab_contents = ["Control", "Analysis"]

    children = [parameters_ui, constraint_ui]
    tab = widgets.Tab()
    tab.children = children
    for i, title in enumerate(tab_contents):
        tab.set_title(i, title)

    display(tab)
    
def display_ex_post_app(list_of_funds,field_to_use):
    date_start = widgets.Text(
        step=1,
        description="Start (YYYY-MM-DD)",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "auto"},
    )

    end_date = widgets.Text(
        step=1,
        description="End (YYYY-MM-DD)",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "auto"},
    )

    def get_data_on_click(b):

        global table

        start = date_start.value
        end = end_date.value

        # Create a date range with end of month frequency
        date_range = pd.date_range(start=start, end=end, freq='ME')

        dico_date={}

        for date in date_range:

            dico_date[get_date_to_string(date)]=date

        tasks = [(get_perf_fields,fund,field,get_date_to_string(date))
            for fund in list_of_funds for field in field_to_use for date in date_range 
        ]

        dico_metrics = {}
        funds_metrics={}
        start_time_sec=time.time()
        with Pool(processes=min(5, cpu_count() * 2)) as pool:
            for date, field, fund,data in pool.imap_unordered(safe_call_fields, tasks):
                temp_date=dico_date[date]
                if not isinstance(data, str):
                    try:
                        if temp_date not in dico_metrics:
                            dico_metrics[temp_date] = {}
                        dico_metrics[temp_date][field] = data['portfolioPerformance'][0]
                        funds_metrics[fund]=pd.DataFrame(dico_metrics).T
                    except Exception as e:

                        print(f"Data not found for fund {fund} for field {field} at date {temp_date}")
        
        finish=time.time()
        
        print(finish-start_time_sec)
        
        table=pd.DataFrame()

        for key in funds_metrics:
            temp=funds_metrics[key]
            temp['Fund']=key
            table=pd.concat([table,temp])

        cols = [c for c in table.columns if c != 'Fund'] + ['Fund']
        table=table[cols]

        dropdown1.options = field_to_use
        dropdown2.options = list_of_funds

        def get_excel(b):

            table.to_excel('Ex Post Data.xlsx', index=True)

            print("File Generated")

        bt_excel = widgets.Button(
            description="Get Excel",
            layout=widgets.Layout(
                display="flex",
                justify_content="center",
                align_items="center",
                spacing="10px",
                width="auto",
            ),
        )

        bt_excel.on_click(get_excel)
        with data_output:
            data_output.clear_output()
            display(display_scrollable_df(table))
            # display(display_scrollable_df(get_perf_catalog()))
            display(bt_excel)
            # display(display_scrollable_df(tables['Data']))


    dropdown1 = widgets.Dropdown(description="Fields:", value=None, options=field_to_use)
    dropdown2 = widgets.Dropdown(description="Funds:", value=None, options=list_of_funds)

    data_output = widgets.Output()
    button_data = widgets.Button(description="Get Data")
    button_data.on_click(get_data_on_click)

    parameters_ui = widgets.VBox(
        [
            widgets.HBox(
                [date_start, end_date, button_data],
                layout=widgets.Layout(
                    display="flex",
                    justify_content="center",
                    align_items="center",
                    spacing="auto",
                    width="auto",
                ),
            ),
            data_output,
        ]
    )

    data = []


    def on_add_constraint_clicked(b):
        row = {"Field": dropdown1.value, "Fund": dropdown2.value}
        data.append(row)
        with constraint_output:
            constraint_output.clear_output()
            display(pd.DataFrame(data))


    add_constraint_btn = widgets.Button(description="Add Filter", button_style="success")
    add_constraint_btn.on_click(on_add_constraint_clicked)

    constraint_output = widgets.Output()
    output = widgets.Output()


    def on_clear_clicked(b):
        data.clear()
        res.clear()
        with constraint_output:
            constraint_output.clear_output()
            display(pd.DataFrame(columns=["Field", "Fund"]))

        with output:
            output.clear_output()


    clear_btn = widgets.Button(description="Clear All", button_style="danger")
    clear_btn.on_click(on_clear_clicked)

    res = {}


    def on_optimize_clicked(b):

        filter_dataframe = pd.DataFrame(data)
        unique_list_funds = set(filter_dataframe["Fund"])
        dico_filter = {}
        for fund in unique_list_funds:
            temp = filter_dataframe[filter_dataframe["Fund"] == fund]
            dico_filter[fund] = list(set(temp["Field"]))

        for key in dico_filter:

            temp = table[table["Fund"] == key][dico_filter[key]]
            res[key] = temp

        with output:
            output.clear_output()
            for key in res:
                display(Markdown("### " + str(key)))
                display(display_scrollable_df(res[key]))


    optimize_btn = widgets.Button(description="Filter", button_style="primary")
    optimize_btn.on_click(on_optimize_clicked)

    constraint_ui = widgets.VBox(
    [
        widgets.VBox([dropdown1, dropdown2]),
        widgets.HBox([add_constraint_btn, clear_btn, optimize_btn]),
        constraint_output,
        output,
    ]
        )

    tab_contents = ["Control", "Analysis"]

    children = [parameters_ui, constraint_ui]
    tab = widgets.Tab()
    tab.children = children
    for i, title in enumerate(tab_contents):
        tab.set_title(i, title)

    display(tab)
    
    

def safe_call(tasks):
    func, ticker, datestart, dateto, type_code = tasks

    try:
        return ticker, func(ticker, datestart, dateto, type_code)
    except Exception as e:
        return {ticker: str(e), "args": tasks}


def display_app(ISINs):
    date_start = widgets.Text(
        step=1,
        description="Start (YYYYMMDD)",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",style={'description_width':'auto'}
    )

    date_end = widgets.Text(
        step=1,
        description="End (YYYYMMDD)",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",style={'description_width':'auto'}
    )

    code_type = widgets.Dropdown(
        options=["DecalogCodePtf", "ISINCodePtf"],
        value="ISINCodePtf",
        description="Code Type",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",style={'description_width':'auto'}
    )

    def get_data_on_click(b):

        global perf_dict, not_found
        tasks = [
            (get_perf, t, date_start.value, date_end.value, code_type.value)
            for t in ISINs
        ]
        perf_dict = {}
        start = time.time()
        not_found = []

        with Pool(processes=min(5, cpu_count() * 2)) as pool:

            for ticker, data in pool.imap_unordered(safe_call, tasks):

                if type(data) is not str:

                    if "benchPerf" in data.columns:
                        data["Excess Returns"] = (
                            data["grossperf"] - data["benchPerf"]
                        ) / 100
                        data = data.rename(
                            columns={
                                "grossBase100": "NAV Base 100",
                                "benchmarkValue": "Benchmark Base 100",
                                "grossperf": "Share Class Return",
                                "benchPerf": "Benchmark Return",
                            }
                        )

                    else:
                        data["benchPerf"] = data["grossperf"].fillna(0) / 100
                        data["Excess Returns"] = (data["grossperf"]) / 100
                        data = data.rename(
                            columns={
                                "grossBase100": "NAV Base 100",
                                "benchmarkValue": "Benchmark Base 100",
                                "grossperf": "Share Class Return",
                                "benchPerf": "Benchmark Return",
                            }
                        )

                    perf_dict[ticker] = data
                else:
                    not_found.append(ticker)

            # results=dict(pool.map(safe_call,tasks))
            # results=pool.map(safe_call,tasks)
        #     print(results)

        #     for r in results:
        #         print(r)

        with data_output:
            data_output.clear_output()
            display(
                display_scrollable_df(pd.DataFrame(not_found, columns=["Missing ETFs"]))
            )
            display(
                display_scrollable_df(
                    pd.DataFrame(list(perf_dict.keys()), columns=["Retrieved ETFs"])
                )
            )

        final = time.time()
        print(f"{final-start:.2f}")

    def get_atypical_perf(b):

        global flagged_cumulative, flagged_daily

        flagged_cumulative = {}
        flagged_daily = {}

        # Daily Control and Cumulative Control on Excess Returns to flag atypical Performance#

        te = {}
        vol = {}
        for key in perf_dict:
            temp = (perf_dict[key]["Excess Returns"]).resample("ME").std() * np.sqrt(
                252
            )
            temp_vol = (perf_dict[key]["Share Class Return"]).resample(
                "ME"
            ).std() * np.sqrt(252)
            te[key] = temp
            vol[key] = temp_vol

        for ISIN in perf_dict:

            temp = perf_dict[ISIN]
            index = np.where(abs(temp["Excess Returns"]) > daily_limit.value / 10000)
            flagged_daily[ISIN] = temp.iloc[index]

            if (
                abs((1 + temp["Excess Returns"]).cumprod().iloc[-1] - 1)
                > cumulative_limit.value / 10000
            ):
                flagged_cumulative[ISIN] = (1 + temp["Excess Returns"]).cumprod()

            else:

                continue

        # Compute the cumulative Excess Returns to be used as a plot
        excess_returns_cumulative = {}
        for key in perf_dict:
            excess_returns_cumulative[key] = (
                1 + perf_dict[key]["Excess Returns"]
            ).cumprod()

        # excess_returns_cumulative_dataframe=pd.DataFrame(excess_returns_cumulative)

        # Get last value of Cumulative Excess Return (to see if above limit and be inserted in a table)

        cumulative = {}
        for ISIN in perf_dict:
            temp = perf_dict[ISIN]
            cumulative[ISIN] = (
                (1 + temp["Excess Returns"]).cumprod().iloc[-1] - 1
            ) * 10000

        # Summary Table for the daily returns and ETFs flagged#

        summary_daily = {}

        for key in flagged_daily:

            try:
                temp = flagged_daily[key]
                count = temp.shape[0]
                max_dev = (temp["Excess Returns"].max() * 10000).round(4)
                min_dev = (temp["Excess Returns"].min() * 10000).round(4)
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

            temp = perf_dict[key]
            wo_dup = temp[~temp.index.duplicated()]
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
                "Date of Max (Downside) Deviation",
            ],
        ).T

        global returns_dataframe, benchmark_returns, excess_returns_dataframe, cumulative_dataframe, monthly_te, monthly_vol

        returns_dataframe = pd.DataFrame(share_class_returns).sort_index()
        benchmark_returns = pd.DataFrame(benchmark).sort_index()
        excess_returns_dataframe = pd.DataFrame(excess).sort_index()

        monthly_te = pd.DataFrame(te).sort_index()
        monthly_vol = pd.DataFrame(vol).sort_index()

        cumulative_dataframe = pd.DataFrame(
            cumulative.values(),
            index=cumulative.keys(),
            columns=["Final Excess Return (Bps)"],
        )
        cumulative_dataframe = cumulative_dataframe.sort_values(
            by="Final Excess Return (Bps)", ascending=False
        )

        fund_list = list(perf_dict.keys())
        
        global selected_fund
        selected_fund=widgets.Dropdown(
        options=fund_list,
        disabled=False,
    )


        def get_excel(b):

            with pd.ExcelWriter(
                "Atypical Performance.xlsx", engine="openpyxl"
            ) as writer:

                returns_dataframe.to_excel(writer, sheet_name="Returns", index=True)
                benchmark_returns.to_excel(writer, sheet_name="Benchmark", index=True)
                excess_returns_dataframe.to_excel(
                    writer, sheet_name="Excess Returns", index=True
                )
                daily_deviation.to_excel(
                    writer, sheet_name="Daily Violations", index=True
                )
                cumulative_dataframe.to_excel(
                    writer, sheet_name="Cumulative Violations", index=True
                )
                monthly_te.to_excel(
                    writer, sheet_name="Monthly Tracking Error in %", index=True
                )
                monthly_vol.to_excel(
                    writer, sheet_name="Monthly Volatility in %", index=True
                )
            
            print("File Generated")
                
        bt_excel = widgets.Button(
            description="Get Excel",
            layout=widgets.Layout(
                display="flex",
                justify_content="center",
                align_items="center",
                spacing="10px",
                width="auto",
            ),
        )

        bt_excel.on_click(get_excel)

        with button_atypical_table:
            button_atypical_table.clear_output()
            display(display_scrollable_df(daily_deviation))
            display(display_scrollable_df(cumulative_dataframe))
            display(bt_excel)
    def get_time_series(value1):

        fund=returns_dataframe[value1]
        bench=benchmark_returns[value1]
        excess=excess_returns_dataframe[value1]

        returns_series=pd.concat([fund,bench,excess],axis=1)
        returns_series.columns=['Fund','Benchmark','Excess Return']

        returns_series=(1+returns_series/100).cumprod()*100

        return returns_series


    def get_monthly_tracking_error(value1):

        excess=excess_returns_dataframe[value1]
        excess.columns=['Monthly Tracking Error in BPS']
        monthly_tracking_error=excess.resample('ME').std()*np.sqrt(252)*100

        return monthly_tracking_error

    def get_monthly_vol(value1):

        returns=returns_dataframe[value1].dropna()
        returns.columns=['Monthly Vol in %']
        monthly_volatility=returns.resample('ME').std()*np.sqrt(252)

        return monthly_volatility

    def plot_chart(b):
        
        
        time_series=get_time_series(selected_fund.value).dropna()
        monthly_te=get_monthly_tracking_error(selected_fund.value).dropna()
        monthly_vol=get_monthly_vol(selected_fund.value).dropna()
        
        plt.style.use("dark_background")
        
        fig=plt.figure()
        plt.plot(time_series[['Fund','Benchmark']])
        plt.title("Cumulative Returns")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.tight_layout()
        
        fig2=plt.figure()
        plt.plot(time_series['Excess Return'])
        plt.title("Cumulative Excess Returns")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.tight_layout()
        
        fig3=plt.figure()
        plt.plot(monthly_te)
        plt.title("Monthly Tracking Error in BPS")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.tight_layout()

        
        
        fig4=plt.figure()
        plt.plot(monthly_vol)
        plt.title("Monthly Vol in %")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.tight_layout()

        with chart_output:
            chart_output.clear_output()
            plt.show()
            display(display_scrollable_df(time_series))
            
    button_data = widgets.Button(description="Get Data")
    data_output = widgets.Output()
    button_data.on_click(get_data_on_click)

    button_atypical = widgets.Button(description="Get Results")
    button_atypical_table = widgets.Output()
    button_atypical.on_click(get_atypical_perf)
    
    selected_fund=widgets.Dropdown(
        options=ISINs,
        disabled=False,
    )
    

    button_chart = widgets.Button(description="Get Chart")
    chart_output = widgets.Output()
    button_chart.on_click(plot_chart)
    
    daily_limit = widgets.BoundedFloatText(
        value=5, step=0.1, description="Daily Limit (BPS)", disabled=False,style={'description_width':'auto'}
    )

    cumulative_limit = widgets.BoundedFloatText(
        value=20, step=0.1, description="Cumulative Limit (BPS)", disabled=False,style={'description_width':'auto'}
    )

    parameters_ui = widgets.VBox(
        [date_start, date_end, code_type, button_data, data_output],
        layout=widgets.Layout(
            display="flex",
            justify_content="center",
            align_items="center",
            spacing="auto",
            width="auto",
        ),
    )

    limit_ui = widgets.VBox([daily_limit, cumulative_limit, button_atypical, button_atypical_table],
                           layout=widgets.Layout(
            display="flex",
            justify_content="center",
            align_items="center",
            spacing="auto",
            width="auto"))

    app = widgets.VBox([parameters_ui, limit_ui])
    chart=widgets.VBox([selected_fund,button_chart,chart_output])
    tab_contents = ["Control","Chart"]
    
    children = [app,chart]
    tab = widgets.Tab()
    tab.children = children
    for i, title in enumerate(tab_contents):
        tab.set_title(i, title)
    display(tab)
    
    
def safe_call_LT(tasks):
    func, date_start, date_end, sessions = tasks

    try:
        return date_start, date_end, func(date_start, date_end, sessions)
    except Exception as e:
        return {date_start: str(e), "args": tasks}


def safe_call_map_exposure(tasks):

    func, df, mapclass, axis = tasks

    try:
        return mapclass, mapclass, axis, func(df, mapclass, axis)
    except Exception as e:
        return {mapclass: str(e), "args": tasks}


def get_LT_exposure(date_start, date_end, alto_smartservices):

    results = alto_smartservices.get(
        query_runner="mapHoldingsExtractLT",
        params={"format": "json", "reportDate": date_start, "reportDate2": date_end},
    )

    return results


def get_map_exposure(dataframe, mapclass="MAPCLASS", axis="EXPOSUREWEIGHT_PTF"):

    ptf_code = list(set(dataframe["SELECTED_FUND_CODE"]))
    map_profile_exposure = {}

    for code in ptf_code:

        ptf = dataframe[dataframe["SELECTED_FUND_CODE"] == code].sort_index().copy()
        ptf = ptf.replace("�", 0)
        map_profil_custom = set(ptf[mapclass])
        ptf[axis] = ptf[axis].astype(float)

        map_profile_exposure[code] = (
            ptf[[axis, mapclass]].groupby(by=[mapclass, ptf.index]).sum()
        )
    return map_profile_exposure

def multi_asset_app(alto_smartservices):
    
    dico_axis={}
    axis = ["EXPOSUREWEIGHT_PTF", "MODIFIED_DURATION_LT_C_PTF"]
    mapclasses = ["MAPCLASS", "MAPZONE1", "MAPZONE2", "MAP_PROFILE_CUSTOM"]
    dico_axis['EXPOSUREWEIGHT_PTF']='Expo'
    dico_axis['MODIFIED_DURATION_LT_C_PTF']='Duration'
    # --- UI WIDGETS ---
    dropdown1 = widgets.Dropdown(description="Funds:", value=None, options=[])
    dropdown2 = widgets.Dropdown(description="Axis:", value=None, options=[])
    dropdown3 = widgets.Dropdown(description="Map Class:", value=None, options=[])

    dropdown1_holding = widgets.Dropdown(description="Funds:", value=None, options=[])
    dropdown2_holding = widgets.Dropdown(description="Axis:", value=None, options=[])
    dropdown3_holding = widgets.Dropdown(description="Map Class:", value=None, options=[])
    dropdown4_holding = widgets.Dropdown(description="Date:", value=None, options=[])

    date_start_widgets = widgets.Text(
        step=1,
        description="Start (YYYY-MM-DD)",
        disabled=False,
        style={"description_width": "auto"},
    )

    end_date_widgets = widgets.Text(
        step=1,
        description="End (YYYY-MM-DD)",
        disabled=False,
        style={"description_width": "auto"},
    )

    # --- CALLBACK TO FETCH DATA ---
    def get_data_on_click(b):
        global dataframe,dataframe_original, dico_map_classes, funds, dates, axis, mapclasses

        # Date range
        date_list = pd.date_range(pd.to_datetime(date_start_widgets.value),pd.to_datetime(end_date_widgets.value),freq="QE",
        )
        date_list = date_list.append(pd.DatetimeIndex([pd.to_datetime(date_start_widgets.value)]))
        date_list =sorted(pd.DatetimeIndex([pd.to_datetime(end_date_widgets.value)]).append(date_list))

        pairs = [(get_date_to_string(date_list[i]), get_date_to_string(date_list[i + 1]))
                 for i in range(len(date_list) - 1)]

        tasks = [(get_LT_exposure, d1, d2, alto_smartservices) for d1, d2 in pairs]

        dico_data = {}
        start = time.time()
        with Pool(processes=min(5, cpu_count() * 2)) as pool:
            for date_start, date_end, data in pool.imap_unordered(safe_call_LT, tasks):
                if not isinstance(data, str):
                    dico_data[date_start] = data
                    print(f"data found at {date_start}")
        print("Data fetch time:", time.time() - start)

        # Merge dataframes
        dataframe_original = pd.concat(dico_data.values(), axis=0)
        dataframe=dataframe_original.copy()

        dataframe = dataframe.replace("�", 0)
        dataframe["HOLDING_DATE"] = pd.to_datetime(dataframe["HOLDING_DATE"],errors="coerce",dayfirst=False,utc=True)
        # dataframe["HOLDING_DATE"]=dataframe["HOLDING_DATE"].dt.tz_localize(None)
        dataframe["HOLDING_DATE"]=dataframe["HOLDING_DATE"].dt.date
        dataframe=dataframe[~dataframe.index.isna()]
        dataframe=dataframe.sort_index()
        dataframe["EXPOSUREWEIGHT_PTF"] = dataframe["EXPOSUREWEIGHT_PTF"].astype(float)
        dataframe["MODIFIEDDURATIONCONTRIB_PTF"] = dataframe["MODIFIEDDURATIONCONTRIB_PTF"].astype(float)
        dataframe = dataframe.drop_duplicates().set_index("HOLDING_DATE")


        # Globals
        dates = sorted(list(set(dataframe.index)))
        funds = list(set(dataframe["SELECTED_FUND_CODE"]))

        # axis_temp = ["EXPOSUREWEIGHT_PTF", "MODIFIED_DURATION_LT_C_PTF"]
        # mapclasses_temp = ["MAPCLASS", "MAPZONE1", "MAPZONE2", "MAP_PROFILE_CUSTOM"]
        axis = ["EXPOSUREWEIGHT_PTF", "MODIFIED_DURATION_LT_C_PTF"]
        mapclasses = ["MAPCLASS", "MAPZONE1", "MAPZONE2", "MAP_PROFILE_CUSTOM"]
        dico_map_classes = {}
        for axis_sum_temp in axis:
            dico_map_classes[axis_sum_temp] = {}
            for maps_temp in mapclasses:
                dico_map_classes[axis_sum_temp][maps_temp] = get_map_exposure(
                    dataframe, mapclass=maps_temp, axis=axis_sum_temp
                )

        # Update dropdowns
        dropdown1.options = funds
        dropdown2.options = axis
        dropdown3.options = mapclasses
        dropdown1_holding.options = funds
        dropdown2_holding.options = axis
        dropdown3_holding.options = mapclasses
        dropdown4_holding.options = dates

        print(f"Processing time: {time.time() - start:.2f} seconds")

    # --- UI BUTTONS ---
    data_output = widgets.Output()
    button_data = widgets.Button(description="Get Data")
    button_data.on_click(get_data_on_click)

    button_data_holding = widgets.Button(description="Get Holding")

    parameters_ui = widgets.VBox([
        widgets.HBox([date_start_widgets, end_date_widgets, button_data]),
        data_output,
    ])

    # --- CONSTRAINT MANAGEMENT ---
    data = []
    data_holding = []
    res = {}
    res_holding = {}

    constraint_output = widgets.Output()
    output = widgets.Output()
    output_filter_holding = widgets.Output()
    holding_output = widgets.Output()

    def on_add_constraint_clicked(b):
        row = {"Fund": dropdown1.value, "Axis": dropdown2.value, "Map Class": dropdown3.value}
        data.append(row)
        with constraint_output:
            constraint_output.clear_output()
            display(pd.DataFrame(data))

    def on_add_filter_clicked(b):
        row = {
            "Date": dropdown4_holding.value,
            "Fund": dropdown1_holding.value,
            "Axis": dropdown2_holding.value,
            "Map Class": dropdown3_holding.value,
        }
        data_holding.append(row)
        with holding_output:
            holding_output.clear_output()
            display(pd.DataFrame(data_holding))

    def on_clear_clicked(b):
        data.clear()
        res.clear()
        with constraint_output:
            constraint_output.clear_output()
            display(pd.DataFrame(columns=["Fund", "Axis", "Map Class"]))
        with output:
            output.clear_output()

    def on_clear_clicked_holding(b):
        data_holding.clear()
        res_holding.clear()
        with holding_output:
            holding_output.clear_output()
            display(pd.DataFrame(columns=["Date", "Fund", "Axis", "Map Class"]))
        with output_filter_holding:
            output_filter_holding.clear_output()

    def on_optimize_clicked(b):
        def get_excel(b):
            with pd.ExcelWriter("Multi Asset Data.xlsx", engine="openpyxl") as writer:
                for key in res:
                    
                    res[key].to_excel(writer, sheet_name=key, index=True)
            
            print("File Generated")

        bt_excel = widgets.Button(
            description="Get Excel",
            layout=widgets.Layout(
                display="flex",
                justify_content="center",
                align_items="center",
                spacing="10px",
                width="auto",
            ),
        )

        bt_excel.on_click(get_excel)
        filter_dataframe = pd.DataFrame(data)
        dico_filter = filter_dataframe.T.to_dict()

        for key, info in dico_filter.items():
            fundname = info["Fund"]
            axis_fund = info["Axis"]
            map_class_fund = info["Map Class"]

            temp = dico_map_classes[axis_fund][map_class_fund][fundname]
            results = pd.concat(
                [temp.loc[idx].rename(columns=lambda c: idx) for idx in temp.index.get_level_values(0).unique()],
                axis=1
            )
            results['Total']=results.sum(axis=1)
            res[f"{fundname} - {dico_axis[axis_fund]} - {map_class_fund}"] = results

        with output:
            output.clear_output()
            for key in res:
                display(Markdown("### " + str(key)))
                display(display_scrollable_df(res[key]))
            display(bt_excel)
            
    def on_show_holding_clicked(b):
        filter_dataframe = pd.DataFrame(data_holding)
        dico_filter = filter_dataframe.T.to_dict()


        for key, info in dico_filter.items():
            date = info["Date"]
            fundname = info["Fund"]
            axis_fund = info["Axis"]
            map_class_fund = info["Map Class"]

            temp1 = dataframe[dataframe.SELECTED_FUND_CODE == fundname].copy()
            temp2 = temp1.loc[date]

            temp_view = axis + [map_class_fund]
            temp_view.insert(0, "ISIN")
            temp_view.insert(1, "BLOOMBERGCODE")
            temp_view.insert(2, "ASSET_MANAGER")

            temp3 = temp2[temp_view].sort_values(by=axis_fund, ascending=False).round(4)
            res_holding[f"{date} - {fundname} - {dico_axis[axis_fund]}"] = temp3

        with output_filter_holding:
            output_filter_holding.clear_output()
            for key in res_holding:
                display(Markdown("### " + str(key)))
                display(display_scrollable_df(res_holding[key]))

    # --- BUTTONS ---
    add_constraint_btn = widgets.Button(description="Add Filter", button_style="success")
    add_constraint_btn.on_click(on_add_constraint_clicked)

    add_constraint_holding_btn = widgets.Button(description="Add Filter", button_style="success")
    add_constraint_holding_btn.on_click(on_add_filter_clicked)

    clear_btn = widgets.Button(description="Clear All", button_style="danger")
    clear_btn.on_click(on_clear_clicked)

    clear_holding_btn = widgets.Button(description="Clear All", button_style="danger")
    clear_holding_btn.on_click(on_clear_clicked_holding)

    optimize_btn = widgets.Button(description="Filter", button_style="primary")
    optimize_btn.on_click(on_optimize_clicked)

    show_holding_button = widgets.Button(description="Filter", button_style="primary")
    show_holding_button.on_click(on_show_holding_clicked)

    # --- UIs ---
    constraint_ui = widgets.VBox([
        widgets.VBox([dropdown1, dropdown2, dropdown3]),
        widgets.HBox([add_constraint_btn, clear_btn, optimize_btn]),
        constraint_output,
        output,
    ])

    holding_ui = widgets.VBox([
        widgets.VBox([dropdown4_holding, dropdown1_holding, dropdown2_holding, dropdown3_holding]),
        widgets.HBox([add_constraint_holding_btn, clear_holding_btn, show_holding_button]),
        holding_output,
        output_filter_holding,
    ])

    tab_contents = ["Control", "Exposure Time Series", "Holdings"]
    children = [parameters_ui, constraint_ui, holding_ui]
    tab = widgets.Tab()
    tab.children = children
    for i, title in enumerate(tab_contents):
        tab.set_title(i, title)

    display(tab)
