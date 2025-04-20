#!/usr/bin/env python
# coding: utf-8

# In[2]:

import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from RiskMetrics import RiskAnalysis, diversification_constraint, create_constraint
from Rebalancing import rebalanced_portfolio, buy_and_hold


def objective(w):
    return np.sqrt(np.sum((w - w0) ** 2))

def sum_equal_one(weight):
    return np.sum(weight) - 1


drop_down_list = []
constraints_options = []
data = pd.DataFrame(columns=["Asset", "Sign", "Limit"])

def load_excel(file):
    global full_matrix, full_matrix_numpy, w0, drop_down_list, drop_down_list_sector, drop_down_list_asset, constraints_options,bounds_sectors_dataframe

    try:
        file = pd.ExcelFile(file.name)
        data_file = file.parse(sheet_name=file.sheet_names)

        holdings = data_file['Holdings'].set_index('Name')
        holdings = holdings.loc[holdings.index != 'Cash EUR']
        holdings['Portfolio Weighting %'] = holdings['Portfolio Weighting %'] / holdings['Portfolio Weighting %'].sum()

        sheets = file.sheet_names
        sheets.remove('Holdings')
        transparency = {}

        for sheet in sheets:
            temp = data_file[sheet].set_index('Name').iloc[:, 1:]
            temp = temp.loc[temp.index != 'Cash EUR']
            temp = temp.loc[holdings.index]
            transparency[sheet] = temp / 100

        full_matrix = pd.DataFrame()
        for key in transparency:
            full_matrix = pd.concat([full_matrix, transparency[key]], axis=1)

        full_matrix_numpy = full_matrix.to_numpy()
        w0 = holdings['Portfolio Weighting %'].loc[full_matrix.index].to_numpy()

        drop_down_list_asset = list(full_matrix.index) + ['All']
        drop_down_list_sector = list(full_matrix.columns)
        drop_down_list = drop_down_list_asset + drop_down_list_sector + [None]
        constraints_options = ["=", "≥", "≤"]
        bounds_sectors={}

        for col in full_matrix.columns:
            min_bounds=round(full_matrix[col].min(),4)
            max_bounds=round(full_matrix[col].max(),4)
        
            bounds_sectors[col]=[min_bounds,max_bounds]

        bounds_sectors_dataframe=pd.DataFrame(bounds_sectors,index=['Lower Bound','Upper Bound']).T.round(4).reset_index().rename(columns={'index': 'Sectors'})
        
        return "File uploaded successfully!",gr.update(choices=drop_down_list)

    except Exception as e:
        return f"Error: {str(e)}", gr.update(choices=[])

def transparency_matrices():

    return full_matrix.reset_index().rename(columns={'index': 'Asset'}).round(4),bounds_sectors_dataframe

def submit(value1, value2, value3):
    global data, constraints

    new_row = {"Asset": value1, "Sign": value2, "Limit": value3}
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    constraint_matrix = pd.DataFrame(data).to_numpy()
    
    constraints = [{'type': 'eq', 'fun': sum_equal_one}]
    
    dico_map = {'=': 'eq', '≥': 'ineq', '≤': 'ineq'}
    
    try:
        for row in range(constraint_matrix.shape[0]):
            temp = constraint_matrix[row, :]
            ticker = temp[0]

            if ticker not in drop_down_list:
                continue

            sign = temp[1]
            limit = float(temp[2])

            if ticker == 'All':
                constraint = diversification_constraint(sign, limit)

            elif ticker in drop_down_list_asset:
                position = np.where(full_matrix.index == ticker)[0][0]
                constraint = create_constraint(sign, limit, position)

            elif ticker in drop_down_list_sector:
                position = np.where(full_matrix.columns == ticker)[0][0]
                if sign == '≤':
                    constraint = [{'type': dico_map[sign], 'fun': lambda weights: limit - (weights @ full_matrix_numpy)[position]}]
                elif sign == '≥':
                    constraint = [{'type': dico_map[sign], 'fun': lambda weights: (weights @ full_matrix_numpy)[position] - limit}]
                else:
                    constraint = [{'type': dico_map[sign], 'fun': lambda weights: (weights @ full_matrix_numpy)[position] - limit}]

            constraints.extend(constraint)

    except Exception as e:
        pass

    return data

def reset_constraints():
    global data, constraints
    data = pd.DataFrame(columns=["Asset", "Sign", "Limit"])
    constraints = []
    return data

def optimize():
    global opt_weights, res
    bounds = [(0, 1) for _ in range(full_matrix.shape[0])]
    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_weights = result.x

    initial = pd.DataFrame(w0, index=full_matrix.index, columns=["Initial"])
    optimal = pd.DataFrame(opt_weights, index=full_matrix.index, columns=["Optimised"])
    res = pd.concat([initial, optimal], axis=1)
    res['Variation']=res['Optimised']-res['Initial']

    # return res, res.T @ full_matrix.round(4).T

    return res.reset_index().rename(columns={'index': 'Asset'}).round(4), (res.T@full_matrix).T.reset_index().rename(columns={'index': 'Sectors'}).round(4)
    
with gr.Blocks(css="* { font-family: 'Arial Narrow', sans-serif; }") as app:    

    with gr.Tab("Rebalancing Optimizer"):
        gr.Markdown("## Rebalancing Optimizer")
    
        # Define UI elements first
        file_input = gr.File(label="Upload Excel File (.xlsx)")
        file_status = gr.Textbox(label="Status", interactive=False)
        asset_dropdown = gr.Dropdown(choices=drop_down_list, label="Asset or Sector")  # << Move this up
        sign_dropdown = gr.Dropdown(choices=["=", "≥", "≤"], label="Sign")
        limit_input = gr.Number(label="Limit (Float)")
    
        file_input.change(load_excel, inputs=file_input, outputs=[file_status, asset_dropdown])
    
        # Button to submit constraint
        submit_button = gr.Button("Add Constraint")
        reset_button = gr.Button("Reset Constraints")
        constraints_table = gr.Dataframe(headers=["Asset", "Sign", "Limit"], interactive=False)
        
        submit_button.click(
            submit,
            inputs=[asset_dropdown, sign_dropdown, limit_input],
            outputs=[constraints_table]
        )
        
        reset_button.click(
            reset_constraints,
            outputs=[constraints_table]
        )
    
    
        # Optimize
        optimize_button = gr.Button("Optimize Portfolio")
        weights_output = gr.Dataframe(label="Optimized Weights")
        exposure_output = gr.Dataframe(label="Optimized Sector Exposure")
    
        optimize_button.click(optimize, outputs=[weights_output,exposure_output])

    with gr.Tab("Transparency Matrix"):
        
        transparency_matrix = gr.Dataframe(headers=["Sectors", "Lower Bound", "Upper Bound"], interactive=False)
        bound_matrix = gr.Dataframe(label="Sectors Bounds")
        
        show_bounds = gr.Button("Underlying Exposure")

        show_bounds.click(transparency_matrices, outputs=[transparency_matrix,bound_matrix])

        
app.launch()


# In[ ]:







