import pandas as pd
import streamlit as st
import plotly.express as px

# def generate_failure_code_analysis(data, fiscal_year, attribute):
#     start_year = int(fiscal_year.split('-')[0])
#     end_year = int(fiscal_year.split('-')[1])

#     start_date = pd.Timestamp(f'{start_year}-04-01')
#     end_date = pd.Timestamp(f'{end_year}-03-31')

#     data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

#     data['Rating'] = data['Rating'].astype(str)

#     if attribute == 'Hours Run':
#         # Bin hours run into predefined slabs
#         bins = [0, 50, 100, 200, 500, 750, 1000, 3000, float('inf')]
#         labels = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']
#         data_current_fy['Hours_Run_Slab'] = pd.cut(data_current_fy['Hours Run'], bins=bins, labels=labels, right=False)
#         attribute = 'Hours_Run_Slab'

#     # Group by Failure Code and the selected attribute
#     failure_code_data = data_current_fy.groupby(['Failure Code', attribute]).size().unstack(fill_value=0)

#     # Calculate row totals
#     failure_code_data['Total'] = failure_code_data.sum(axis=1)
#     failure_code_data.loc['Total'] = failure_code_data.sum()

#     return failure_code_data

import pandas as pd
import streamlit as st
import plotly.express as px

# Function to generate failure code analysis
def generate_failure_code_analysis(data, fiscal_year, attribute, customer_groups,selected_oems,selected_months):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    if selected_months:
            data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')
            data_current_fy = data_current_fy[data_current_fy['Month'].isin(selected_months)]

    # Filter data by selected customer groups
    if customer_groups:
        data_current_fy = data_current_fy[data_current_fy['Customer Group'].isin(customer_groups)]

    if selected_oems:
        data_current_fy = data_current_fy[data_current_fy['Supplier/OEM'].isin(selected_oems)]

    if attribute == 'Hours Run':
        bins = [0, 50, 100, 200, 500, 750, 1000, 3000, float('inf')]
        labels = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']
        data_current_fy['Hours_Run_Slab'] = pd.cut(data_current_fy['Hours Run'], bins=bins, labels=labels, right=False)
        attribute = 'Hours_Run_Slab'
    else:
        data_current_fy[attribute] = data_current_fy[attribute].astype(str)

    failure_code_data = data_current_fy.groupby(['Failure Code', attribute]).size().unstack(fill_value=0)

    failure_code_data['Total'] = failure_code_data.sum(axis=1)
    failure_code_data.loc['Total'] = failure_code_data.sum()

    return failure_code_data

# Function to plot failure code analysis

def plot_failure_code_analysis(data, fiscal_year, attribute, selected_values, customer_groups, selected_oems, selected_months):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    if selected_months:
        data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')
        data_current_fy = data_current_fy[data_current_fy['Month'].isin(selected_months)]

    # Filter data by selected customer groups
    if customer_groups:
        data_current_fy = data_current_fy[data_current_fy['Customer Group'].isin(customer_groups)]

    if selected_oems:
        data_current_fy = data_current_fy[data_current_fy['Supplier/OEM'].isin(selected_oems)]

    if attribute == 'Hours Run':
        bins = [0, 50, 100, 200, 500, 750, 1000, 3000, float('inf')]
        labels = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']
        data_current_fy['Hours_Run_Slab'] = pd.cut(data_current_fy['Hours Run'], bins=bins, labels=labels, right=False)
        attribute = 'Hours_Run_Slab'
    else:
        data_current_fy[attribute] = data_current_fy[attribute].astype(str)

    failure_code_data = data_current_fy.groupby(['Failure Code', attribute]).size().reset_index(name='Count')

    if selected_values:
        selected_values = [str(val) for val in selected_values]
        failure_code_data = failure_code_data[failure_code_data[attribute].isin(selected_values)]

    if attribute == 'Hours_Run_Slab':
        category_order = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']
        failure_code_data[attribute] = pd.Categorical(failure_code_data[attribute], categories=category_order, ordered=True)
        failure_code_data = failure_code_data.sort_values(by=attribute)
    elif attribute == 'Rating':
        failure_code_data[attribute] = pd.Categorical(failure_code_data[attribute])

    fig = px.bar(
        failure_code_data,
        x='Failure Code',
        y='Count',
        color=attribute,
        title=f"Failure Code Analysis for {fiscal_year}",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        text='Count'  # Add text values on the bars
    )

    fig.update_layout(
        xaxis_title='Failure Code',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        title=f"Failure Code Analysis for {fiscal_year}",
        showlegend=True,
        legend_title=attribute,
        margin=dict(t=40, b=40, l=40, r=40)  # Adjust margins to provide more space
    )

    fig.update_traces(
        texttemplate='%{text}',  # Format text to 2 decimal places
        textposition='inside',  # Place text inside the bars
        insidetextanchor='middle',  # Center text inside the bars
        textfont=dict(size=12, color='white'),  # Adjust font size and color
        marker=dict(line=dict(color='rgb(0,0,0)', width=1))  # Optional: Add border to bars
    )

    st.plotly_chart(fig)



# def plot_failure_code_analysis(data, fiscal_year, attribute, selected_values):
#     start_year = int(fiscal_year.split('-')[0])
#     end_year = int(fiscal_year.split('-')[1])

#     start_date = pd.Timestamp(f'{start_year}-04-01')
#     end_date = pd.Timestamp(f'{end_year}-03-31')

#     data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

#     if attribute == 'Hours Run':
#         # Bin hours run into predefined slabs
#         bins = [0, 50, 100, 200, 500, 750, 1000, 3000, float('inf')]
#         labels = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']
#         data_current_fy['Hours_Run_Slab'] = pd.cut(data_current_fy['Hours Run'], bins=bins, labels=labels, right=False)
#         attribute = 'Hours_Run_Slab'
    
#     data_current_fy[attribute] = data_current_fy[attribute].astype(str)

#     # Group by Failure Code and the selected attribute
#     failure_code_data = data_current_fy.groupby(['Failure Code', attribute]).size().reset_index(name='Count')

#     # Filter data based on selected values
#     if selected_values:
#         failure_code_data = failure_code_data[failure_code_data[attribute].isin(selected_values)]

#     # Sort the data by the attribute to maintain order
#     if attribute == 'Hours_Run_Slab':
#         category_order = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']
#         failure_code_data[attribute] = pd.Categorical(failure_code_data[attribute], categories=category_order, ordered=True)
#         failure_code_data = failure_code_data.sort_values(by=attribute)

#     # Create a plot
#     fig = px.bar(
#         failure_code_data,
#         x='Failure Code',
#         y='Count',
#         color=attribute,
#         title=f"Failure Code Analysis for {fiscal_year}",
#         color_discrete_sequence=px.colors.qualitative.Plotly  # Use distinct colors for categorical data
#     )

#     # Sort x-axis categories by attribute if Hours_Run_Slab
#     if attribute == 'Hours_Run_Slab':
#         fig.update_layout(
#             xaxis=dict(
#                 categoryorder='array',
#                 categoryarray=category_order
#             )
#         )

#     st.plotly_chart(fig)
