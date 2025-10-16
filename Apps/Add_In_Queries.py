#!/usr/bin/env python
# coding: utf-8

# In[117]:


from suds.client import Client
import numpy as np
import json
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt


# # Portfolio Returns

# In[118]:
def get_date_to_string(date):
    yyyy = date.year
    mm = date.month
    dd = date.day

    if mm < 10:

        mm = "0" + str(mm)

    if dd < 10:

        dd = "0" + str(dd)

    date_to_string = str(yyyy) + str(mm) + str(dd)
    return date_to_string


def get_perf(ISIN, dateFrom, dateTo,type_code='ISINCodePtf'):

    url = f"https://perfo-api.intramundi.com/performance/api/v1/valuationData/flatten?typeCode={type_code}&assetcode={ISIN}&dateFrom={dateFrom}&dateTo={dateTo}"
    # json_data = requests.get(url)

    # data=pd.read_json(json_data.text)
    
    data=pd.read_json(url)
    data["valuationDate"] = pd.to_datetime(
        data["valuationDate"], errors="coerce", utc=False
    )
    data["valuationDate"] = data["valuationDate"].apply(
        lambda x: x.replace(tzinfo=None)
    )
    data = data.set_index("valuationDate")
    # data.index=data.index.normalize()

    return data

def get_perf_fields(decalog,label, dateFrom="", dateTo="",type_code="DecalogCodePtf",currency="EUR",type_part="C"):

    url = f"https://perfo-api.intramundi.com/performance/api/v1/reporting/performance/performances?ip=selvam&assetcode={decalog}&typecode={type_code}&assetsharecode={type_part}&codereqperf={label}&currency={currency}&dateFrom={dateFrom}&dateTo={dateTo}&indiceType=(%27BENCHMARK%27)"

    data = pd.read_json(url)
    return data

def get_perf_catalog():
    url="https://perfo-api.intramundi.com/performance/api/v1/reporting/performance/catalog"

    data = pd.read_json(url)
    return data

# # Index Valuation

# In[119]:


def get_index_valuation(index,start_date,end_date,currency='EUR'):
    
    wsdl = "https://platf-api.intramundi.com/mediaplus-ws/services//IndexExporter/IndexExporter?wsdl"
    client = Client(wsdl)
    response = client.service.getValoIndiceDateRange(in0=index,in1=start_date, in2=end_date,in3=currency)

    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]

        for j in range(len(current_row)):
            row.append(current_row[j][0])

        data.append(row)

    return pd.DataFrame(data).replace('soapenc:string',0)


# # Index Composition

# In[120]:


def retrieve_index_composition(index,date):
    
    wsdl = 'https://platf-api.intramundi.com/mediaplus-ws/services//IndexExporter/IndexExporter?wsdl'
    client = Client(wsdl)
    response = client.service.getCompoIndiceFilterCash(index, date)
    
    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]
    
        for j in range(len(current_row)):
            row.append(current_row[j][0])
    
        data.append(row)
        
    return pd.DataFrame(data).replace('soapenc:string',0)
        


# # ESG Score of an Instrument

# In[6]:


def get_instrument_esg_score(instrument_code,universe,date,in3=False,in4=False):

    wsdl = "https://platf-api.intramundi.com/mediaplus-ws/services//IssuerRatingsExporter/IssuerRatingsExporter?wsdl"
    client = Client(wsdl)
    array_of_string_1 = client.factory.create("ns0:arrayOfString")
    array_of_string_1.i = instrument_code

    # columnsRaw = client.service.getInstrumentFieldsList()
    # columns = [x["i"][0]["value"] for x in columnsRaw["item"]]
    array_of_string_2 = client.factory.create("ns0:arrayOfString")
    array_of_string_2.i =universe

    response = client.service.getIssuerRatingsESG(
        in0=array_of_string_1, in1=array_of_string_2, in2=date, in3=in3, in4=in4
    )
    
    
    
    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]
    
        for j in range(len(current_row)):
            row.append(current_row[j][0])
    
        data.append(row)
        
    return pd.DataFrame(data).replace('soapenc:string',0)
        


# # Instrument Quotes

# In[12]:


def get_instrument_quote(instrument_code,start_date,end_date):
    
    wsdl='https://platf-api.intramundi.com/mediaplus-ws/services//QuoteExporter/QuoteExporter?wsdl'
    client = Client(wsdl)
    
    array_of_string_1 = client.factory.create('ns0:arrayOfString')
    array_of_string_1.i = [instrument_code]
    response=client.service.getQuoteInstrumentDateRange(in0=array_of_string_1, in1=start_date,in2=end_date)
    
    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]

        for j in range(len(current_row)):
            row.append(current_row[j][0])

        data.append(row)
    
    return pd.DataFrame(data).replace('soapenc:string',0)


# # Portfolio Positions

# In[14]:


# response


# In[15]:


def get_positions(decalog_code,date,in3=False,
                  in4=False,
                  in5=True,
                  in6=True,
                  in7=False,
                  in8=False,
                  in9='RM',
                  in10=''):
    
    wsdl='https://platf-api.intramundi.com/mediaplus-ws/services//PortfolioExporter/PortfolioExporter?wsdl'
    client = Client(wsdl)

    array_of_string_1 = client.factory.create('ns0:arrayOfString')
    array_of_string_1.i = [decalog_code]

    columnsRaw = client.service.getPositionFieldsList()
    columns = [x["i"][0]["value"] for x in columnsRaw["item"]]
    array_of_string_2 = client.factory.create('ns0:arrayOfString')
    array_of_string_2.i = [columns]
    
    
    response=client.service.getPositionByFields(in0=array_of_string_2, in1=array_of_string_1,
                                  in2=date,
                                 in3=in3,
                                 in4=in4,
                                 in5=in5,
                                 in6=in6,
                                 in7=in7,
                                 in8=in8,
                                in9=in9,
                                 in10=in10)
    
    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]

        for j in range(len(current_row)):
            row.append(current_row[j][0])

        data.append(row)
    
    return pd.DataFrame(data,columns=columns).replace('soapenc:string',0)


# # Instrument Data

# In[74]:


def get_instrument_data(instrument_code,date):
    
    wsdl='https://platf-api.intramundi.com/mediaplus-ws/services//InstrumentExporter/InstrumentExporter?wsdl'
    client = Client(wsdl)

    array_of_string_1 = client.factory.create('ns0:arrayOfString')
    array_of_string_1.i = [instrument_code]

    columnsRaw = client.service.getInstrumentFieldsList()
    columns = [x["i"][0]["value"] for x in columnsRaw["item"]]
    array_of_string_2 = client.factory.create('ns0:arrayOfString')
    array_of_string_2.i = [columns]
    array_of_string_3 = client.factory.create('ns0:arrayOfString')

    response=client.service.getInstrumentByFieldsForInstNum(in0=array_of_string_2, in1=array_of_string_1,
                                  in2=date,in3="RM",in4=array_of_string_3,in5=False)
    
    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]

        for j in range(len(current_row)):
            row.append(current_row[j][0])

        data.append(row)
    
    return pd.DataFrame(data,columns=columns).replace('soapenc:string',0)


def get_instrument_data_ISIN(instrument_code,date):
    
    wsdl='https://platf-api.intramundi.com/mediaplus-ws/services//InstrumentExporter/InstrumentExporter?wsdl'
    client = Client(wsdl)

    array_of_string_1 = client.factory.create('ns0:arrayOfString')
    array_of_string_1.i = [instrument_code]

    columnsRaw = client.service.getInstrumentFieldsList()
    columns = [x["i"][0]["value"] for x in columnsRaw["item"]]
    array_of_string_2 = client.factory.create('ns0:arrayOfString')
    array_of_string_2.i = [columns]
    array_of_string_3 = client.factory.create('ns0:arrayOfString')

    response=client.service.getInstrumentByFields(in0=array_of_string_2, in1=array_of_string_1,
                                  in2='ISIN',in3=date,in4="RM",in5=array_of_string_3,in6=False)
    
    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]

        for j in range(len(current_row)):
            row.append(current_row[j][0])

        data.append(row)
    
    return pd.DataFrame(data,columns=columns).replace('soapenc:string',0)


# # Calculate ESG score of an Indice

# In[387]:


def get_indice_esg_data(indice,date):

    indice=retrieve_index_composition(indice,date)
    ISIN_list=indice[4].to_list()

    instrument_data=get_instrument_data_ISIN(ISIN_list,date)
    isin_set=set(ISIN_list)
    retrieved_set=set(instrument_data['instCodeIsin'])
    instrument_data=instrument_data.set_index('instCodeIsin')

    dict_issuer=instrument_data['instIssuerCode'].to_dict()
    indice['Issuer Code']=indice[4].replace(dict_issuer)

    issuer_codes=list(set(indice['Issuer Code']))
    esg_scores_issuers=get_instrument_esg_score(issuer_codes,'716',date)
    
    esg_set=set(esg_scores_issuers[11])
    asset_not_retrieved=list(set(issuer_codes).difference(set(esg_set)))
    indice.loc[~indice['Issuer Code'].isin(asset_not_retrieved)]
    esg_scores_dict=esg_scores_issuers.set_index(11)[7].to_dict()
    
    indice['ESG Score']=indice['Issuer Code'].replace(esg_scores_dict)
    indice['ESG Score']=indice['ESG Score'].astype(float)
    indice[2]=indice[2].astype(float)

    temp=indice.loc[~indice['Issuer Code'].isin(asset_not_retrieved)].copy()
    temp[2]=temp[2]/np.sum(temp[2])

    temp['Weighted ESG Score']=temp['ESG Score']*temp[2]
    esg_table=temp[[4,1,2,'Issuer Code','ESG Score','Weighted ESG Score']]
    esg_table.columns=['ISIN','Name','Weight','Issuer Code','ESG Score','ESG Score Contribution']
    esg_score=temp['Weighted ESG Score'].sum()


    
    return esg_score, esg_table,asset_not_retrieved


def retrieve_bench_composition(ptf,date):

    wsdl = 'https://platf-api.intramundi.com/mediaplus-ws/services//IndexExporter/IndexExporter?wsdl'
    client = Client(wsdl)
    response = client.service.getBenchmark(ptf,'' ,date)

    data=[]

    for i in range(len(response['item'])):
        current_row=response["item"][i]["i"]
        row=[]

        for j in range(len(current_row)):
            row.append(current_row[j][0])

        data.append(row)

    return pd.DataFrame(data).replace('soapenc:string',0)
