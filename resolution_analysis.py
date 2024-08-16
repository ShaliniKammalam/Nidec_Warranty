import pandas as pd
import streamlit as st
import plotly.express as px

def calculate_percentage_table(analysis_data):
    percentage_data = analysis_data.copy()
    for index, row in percentage_data.iterrows():
        row_total = row['Total']
        if row_total != 0:  # To avoid division by zero
            for column in percentage_data.columns[1:-1]:  # Exclude 'Created By' and 'Total' columns
                percentage_data.at[index, column] = (row[column] / row_total) * 100
    percentage_data = percentage_data.round(2)
    return percentage_data

# def generate_resolution_analysis(data, fiscal_year, status_filter, selected_months):
#     start_year = int(fiscal_year.split('-')[0])
#     end_year = int(fiscal_year.split('-')[1])

#     start_date = pd.Timestamp(f'{start_year}-04-01')
#     end_date = pd.Timestamp(f'{end_year}-03-31')

#     data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]
    
#     if status_filter == 'Open':
#         # Filter data for 'Open' status
#         #data_status = data_current_fy[data_current_fy['Status'] == 'Open']
#         data_status = data_current_fy[data_current_fy['Is the Complaint Closed'] == 'No']


#         data_status['Month'] = data_status['Complaint Date'].dt.to_period('M').astype(str)
        
#         # Create a pivot table with 'Created By' as rows and 'Month' as columns
#         analysis_data = data_status.pivot_table(index='Created By', columns='Month', aggfunc='size', fill_value=0).reset_index()
        
#         # Add row totals
#         analysis_data['Total'] = analysis_data.drop('Created By', axis=1).sum(axis=1)
        
#         # Add column totals
#         col_totals = analysis_data.drop('Created By', axis=1).sum()
#         col_totals.name = 'Total'
#         col_totals = pd.DataFrame([col_totals], index=['Total'])
        
#         # Concatenate the column totals DataFrame
#         analysis_data = pd.concat([analysis_data, col_totals])
    
#     elif status_filter == 'Released':
#         # Filter data for 'Released' status
#         #data_status = data_current_fy[data_current_fy['Status'] == 'Released']
#         data_status = data_current_fy[data_current_fy['Is the Complaint Closed'] == 'Yes']

#         if selected_months:
#             data_status['Month'] = data_status['Complaint Date'].dt.to_period('M').astype(str)
#             data_status = data_status[data_status['Month'].isin(selected_months)]
        
#         if 'Range of Leadtime' not in data_status.columns:
#             raise ValueError("The column 'Range of Leadtime' is missing in the dataset.")
        
#         # Create a pivot table with 'Created By' as rows and 'Range of Leadtime' as columns
#         analysis_data = data_status.pivot_table(index='Created By', columns='Range of Leadtime', aggfunc='size', fill_value=0).reset_index()
        
#         # Reorder the columns as desired
#         desired_order = ['24 Hrs', '2-5 Days', '6-10 Days', '>10 Days']
#         columns_order = [col for col in desired_order if col in analysis_data.columns]
#         #columns_order.append('Total')  # Ensure 'Total' is at the end
        
#         if 'Created By' in analysis_data.columns:
#             analysis_data = analysis_data[['Created By'] + columns_order]
#         else:
#             analysis_data = analysis_data[columns_order]
        

#         # Add row totals
#         analysis_data['Total'] = analysis_data.drop('Created By', axis=1).sum(axis=1)
        
#         # Add column totals
#         col_totals = analysis_data.drop('Created By', axis=1).sum()
#         col_totals.name = 'Total'
#         col_totals = pd.DataFrame([col_totals], index=['Total'])
        
#         # Concatenate the column totals DataFrame
#         analysis_data = pd.concat([analysis_data, col_totals])

#         # Calculate the percentage of complaints for each row
#         total_complaints_all = analysis_data.loc['Total', 'Total']
#         analysis_data['% of Complaints'] = (analysis_data['Total'] / total_complaints_all * 100).round(2)

#     return analysis_data



def generate_resolution_analysis(data, fiscal_year, status_filter, selected_months):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    if selected_months:
        data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')
        data_current_fy = data_current_fy[data_current_fy['Month'].isin(selected_months)]
    else:
        data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')

    if status_filter == 'Open':
        data_status = data_current_fy[data_current_fy['Is the Complaint Closed'] == 'No']
        analysis_data = data_status.pivot_table(index='Zonal In-Charge', columns='Month', aggfunc='size', fill_value=0).reset_index()

        # Extract month names and sort them
        month_order = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS').strftime('%b-%y').tolist()
        month_order = [month for month in month_order if month in analysis_data.columns]
        
        # Reorder columns based on sorted month order
        analysis_data = analysis_data[['Zonal In-Charge'] + month_order]
        analysis_data['Total'] = analysis_data.drop('Zonal In-Charge', axis=1).sum(axis=1)
        col_totals = analysis_data.drop('Zonal In-Charge', axis=1).sum()
        col_totals.name = 'Total'
        col_totals = pd.DataFrame([col_totals], index=['Total'])
        analysis_data = pd.concat([analysis_data, col_totals])

    elif status_filter == 'Released':
        data_status = data_current_fy[data_current_fy['Is the Complaint Closed'] == 'Yes']

        if selected_months:
            data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')
            data_current_fy = data_current_fy[data_current_fy['Month'].isin(selected_months)]

        if 'Range of Leadtime' not in data_status.columns:
            raise ValueError("The column 'Range of Leadtime' is missing in the dataset.")
        
        analysis_data = data_status.pivot_table(index='Zonal In-Charge', columns='Range of Leadtime', aggfunc='size', fill_value=0).reset_index()
        
        desired_order = ['24 Hrs', '2-5 Days', '6-10 Days', '>10 Days']
        columns_order = [col for col in desired_order if col in analysis_data.columns]
        if 'Zonal In-Charge' in analysis_data.columns:
            analysis_data = analysis_data[['Zonal In-Charge'] + columns_order]
        else:
            analysis_data = analysis_data[columns_order]
        
        analysis_data['Total'] = analysis_data.drop('Zonal In-Charge', axis=1).sum(axis=1)
        col_totals = analysis_data.drop('Zonal In-Charge', axis=1).sum()
        col_totals.name = 'Total'
        col_totals = pd.DataFrame([col_totals], index=['Total'])
        analysis_data = pd.concat([analysis_data, col_totals])

        total_complaints_all = analysis_data.loc['Total', 'Total']
        analysis_data['% of Complaints'] = (analysis_data['Total'] / total_complaints_all * 100).round(2)

    elif status_filter == 'All' or status_filter is None:
        if selected_months:
            data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')
            data_current_fy = data_current_fy[data_current_fy['Month'].isin(selected_months)]
        
        if 'Range of Leadtime' not in data_current_fy.columns:
            raise ValueError("The column 'Range of Leadtime' is missing in the dataset.")
        
        analysis_data = data_current_fy.pivot_table(index='Zonal In-Charge', columns='Range of Leadtime', aggfunc='size', fill_value=0).reset_index()
        
        desired_order = ['24 Hrs', '2-5 Days', '6-10 Days', '>10 Days', 'Open']
        columns_order = [col for col in desired_order if col in analysis_data.columns]
        if 'Zonal In-Charge' in analysis_data.columns:
            analysis_data = analysis_data[['Zonal In-Charge'] + columns_order]
        else:
            analysis_data = analysis_data[columns_order]
        
        analysis_data['Total'] = analysis_data.drop('Zonal In-Charge', axis=1).sum(axis=1)
        col_totals = analysis_data.drop('Zonal In-Charge', axis=1).sum()
        col_totals.name = 'Total'
        col_totals = pd.DataFrame([col_totals], index=['Total'])
        analysis_data = pd.concat([analysis_data, col_totals])

        total_complaints_all = analysis_data.loc['Total', 'Total']
        analysis_data['% of Complaints'] = (analysis_data['Total'] / total_complaints_all * 100).round(2)

    return analysis_data





def plot_resolution_analysis(analysis_data, status_filter):
    x_axis_col = 'Month'
    title = "MTTR by Employee"
    
    if status_filter == 'Open':
        title = "Open Status Count by Month"
        x_axis_col = 'Month'
        
    elif status_filter == 'Released':
        title = "Released Status Count by Range of Leadtime"
        x_axis_col = 'Range of Leadtime'

    columns_to_drop = ['Total', '% of Complaints']
    existing_columns_to_drop = [col for col in columns_to_drop if col in analysis_data.columns]
    
    # Drop unnecessary columns
    plotting_data = analysis_data.drop(columns=existing_columns_to_drop, errors='ignore')
    
    if plotting_data.empty:
        st.write("No data available for the selected filters.")
        return

    # Melt the dataframe for plotting
    melted_data = plotting_data.melt(id_vars='Zonal In-Charge', var_name=x_axis_col, value_name='Count')
    
    # fig = px.bar(
    #     melted_data,
    #     x='Created By',
    #     y='Count',
    #     color=x_axis_col,
    #     title=title,
    #     color_discrete_sequence=px.colors.qualitative.Plotly,
    #     text='Count'  # Add text values on the bars
    # )
    
    # fig.update_layout(
    #     xaxis_title='Created By',
    #     yaxis_title='Count',
    #     xaxis=dict(type='category')  # Ensure x-axis treats categories as strings
    # )

    fig = px.bar(
    melted_data,
    x='Zonal In-Charge',
    y='Count',
    color=x_axis_col,
    title=title,
    color_discrete_sequence=px.colors.qualitative.Plotly,
    text='Count'  # Add text values on the bars
    )

    fig.update_layout(
        xaxis_title='Zonal In-Charge',
        yaxis_title='Count',
        xaxis=dict(type='category'),  # Ensure x-axis treats categories as strings
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        title=title,  # Use the provided title
        showlegend=True,
        legend_title=x_axis_col,  # Title for the legend based on color
        margin=dict(t=40, b=40, l=40, r=40)  # Adjust margins for better spacing
    )

    fig.update_traces(
        texttemplate='%{text}',  # Keep the text formatting consistent
        textposition='inside',  # Place the text inside the bars
        insidetextanchor='middle',  # Center the text within the bars
        textfont=dict(size=12, color='white'),  # Set font size and color
        marker=dict(line=dict(color='rgb(0,0,0)', width=1))  # Optional: Add border to bars
    )

    
    st.plotly_chart(fig)








def generate_customer_group_analysis(data, fiscal_year, selected_names,selected_months):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    # Filter data by selected name
    data_filtered = data_current_fy[data_current_fy['Zonal In-Charge'].isin(selected_names)]


    # Check if 'Range of Leadtime' column exists
    if 'Range of Leadtime' not in data_filtered.columns:
        raise ValueError("The column 'Range of Leadtime' is missing in the dataset.")
    
    if selected_months:
        data_filtered['Month'] = data_filtered['Complaint Date'].dt.strftime('%b-%y')
        data_filtered = data_filtered[data_filtered['Month'].isin(selected_months)]

    # Create a pivot table with 'Customer Group' as rows and 'Range of Leadtime' as columns
    analysis_data = data_filtered.pivot_table(index='Customer Group', columns='Range of Leadtime', aggfunc='size', fill_value=0).reset_index()

    desired_order = ['24 Hrs', '2-5 Days', '6-10 Days', '>10 Days',"Open"]  # Adjust as needed

    # Reorder columns based on the desired order
    ordered_columns = [col for col in desired_order if col in analysis_data.columns]  # Ensure only existing columns are included
    #ordered_columns.append('Total')  # Add 'Total' at the end
    ordered_columns = ['Customer Group'] + ordered_columns  # Ensure 'Customer Group' is first

    analysis_data = analysis_data[ordered_columns]  # Reorder columns

    # Add row totals
    analysis_data['Total'] = analysis_data.drop('Customer Group', axis=1).sum(axis=1)
    
    # Add column totals
    col_totals = analysis_data.drop('Customer Group', axis=1).sum()
    col_totals.name = 'Total'
    col_totals = pd.DataFrame([col_totals], index=['Total'])
    
    # Concatenate the column totals DataFrame
    analysis_data = pd.concat([analysis_data, col_totals])

    # Calculate the percentage of complaints for each row
    total_complaints_all = analysis_data.loc['Total', 'Total']
    analysis_data['% of Complaints'] = (analysis_data['Total'] / total_complaints_all * 100).round(2)

    return analysis_data


def plot_customer_group_analysis(analysis_data):
    # Melt the dataframe for plotting
    plotting_data = analysis_data.drop(columns=['Total', '% of Complaints'])

    # Melt the dataframe for plotting
    melted_data = plotting_data.melt(id_vars='Customer Group', var_name='Range of Leadtime', value_name='Count')
    
    # Plot the data
    # fig = px.bar(
    #     melted_data,
    #     x='Customer Group',
    #     y='Count',
    #     color='Range of Leadtime',
    #     title="Customer Group Analysis by Range of Leadtime",
    #     color_discrete_sequence=px.colors.qualitative.Plotly,
    #     text='Count'  # Add text values on the bars
    # )
    
    # fig.update_layout(
    #     xaxis_title='Customer Group',
    #     yaxis_title='Count',
    #     xaxis=dict(type='category')  # Ensure x-axis treats categories as strings
    # )
    

    fig = px.bar(
    melted_data,
    x='Customer Group',
    y='Count',
    color='Range of Leadtime',
    title="Customer Group Analysis by Range of Leadtime",
    color_discrete_sequence=px.colors.qualitative.Plotly,
    text='Count'  # Add text values on the bars
    )

    fig.update_layout(
        xaxis_title='Customer Group',
        yaxis_title='Count',
        xaxis=dict(type='category'),  # Ensure x-axis treats categories as strings
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        showlegend=True,
        legend_title='Range of Leadtime',  # Title for the legend based on color
        margin=dict(t=40, b=40, l=40, r=40)  # Adjust margins for better spacing
    )

    fig.update_traces(
        texttemplate='%{text}',  # Keep the text formatting consistent
        textposition='inside',  # Place the text inside the bars
        insidetextanchor='middle',  # Center the text within the bars
        textfont=dict(size=12, color='white'),  # Set font size and color
        marker=dict(line=dict(color='rgb(0,0,0)', width=1))  # Optional: Add border to bars
    )

    st.plotly_chart(fig)

