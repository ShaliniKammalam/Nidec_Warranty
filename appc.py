import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import matplotlib.colors as mcolors
from failure_analysis import generate_failure_code_analysis, plot_failure_code_analysis
from resolution_analysis import generate_resolution_analysis,plot_resolution_analysis,calculate_percentage_table,generate_customer_group_analysis,plot_customer_group_analysis


frame_mapping = {
    'LA': 'LSA20',
    'LB': 'LSA423',
    'LC': 'LSA442',  # Keeping only one 'LC' mapping; prioritize if needed
    'LD': 'LSA462',
    'LE': 'LSA472',
    'LF': 'LSA491',  # Keeping only one 'LF' mapping; prioritize if needed
    'LG': 'LSA502',
    'LJ': 'LSA523',
    'LK': 'LSA532',
    'LM': 'LSA542',  # Keeping only one 'LM' mapping; prioritize if needed
}


def map_frame_codes(data):
    # Function to get the frame code based on machine number and existing frame code
    def get_frame_code(machine_no, frame_code):
        if pd.notna(frame_code):
            return frame_code
        
        if pd.notna(machine_no):
            # Extract the first two characters of machine number and match with frame mapping
            machine_prefix = machine_no[:2]
            return frame_mapping.get(machine_prefix, np.nan)
        
        return np.nan  # Return NaN if no match is found

    # Apply the function to the DataFrame
    data['Frame'] = data.apply(lambda row: get_frame_code(row['Machine No.'], row['Frame']), axis=1)

    return data


def get_item_code(df, selected_description):
    """
    Retrieves the item code based on the selected item description.
    """
    try:
        # Ensure the 'Item Description' and 'Item Code' columns exist
        if 'Item Description' not in df.columns or 'Item Code' not in df.columns:
            st.error("Data does not contain required columns 'Item Description' or 'Item Code'")
            return None
        
        # Filter the dataframe based on the selected description
        item_code = df[df['Item Description'] == selected_description]['Item Code'].unique()
        
        if len(item_code) > 0:
            return item_code[0]  # Return the first match (assuming descriptions are unique)
        else:
            st.warning(f"No item code found for description: {selected_description}")
            return None
    
    except Exception as e:
        st.error(f"Error retrieving item code: {e}")
        return None
    

def extract_fiscal_years_foc(data):
    """
    Extracts fiscal years from the 'Date' column.
    Fiscal year is defined from April to March of the following year.
    """
    fiscal_years = data['Date'].dt.to_period('Q-APR').dt.year.unique()
    fiscal_years.sort()  # Sort in ascending order
    fiscal_years = [f"{year}-{year + 1}" for year in fiscal_years]
    fiscal_years.reverse()  # Reverse to get descending order
    return fiscal_years

def generate_customer_item_table(df, selected_fy, selected_months, selected_customers, selected_items):
    """
    Generates a table with rows as customer names, columns as item codes,
    and values as the count of each item code for each customer.
    Filters based on selected fiscal year, months, customers, and item codes.
    """
    try:
        # Ensure 'Customer Name' and 'Item Code' columns exist
        if 'Customer Name' not in df.columns or 'Item Code' not in df.columns:
            st.error("Data does not contain required columns 'Customer Name' or 'Item Code'")
            return pd.DataFrame()
        
        # Filter data based on fiscal year and months
        start_year = int(selected_fy.split('-')[0])
        end_year = int(selected_fy.split('-')[1])
        start_date = pd.Timestamp(f'{start_year}-04-01')
        end_date = pd.Timestamp(f'{end_year}-03-31')
        
        df['Date'] = pd.to_datetime(df['Date'])
        filtered_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        if selected_months:
            months = [pd.Timestamp(f'{start_year}-{month}-01') for month in selected_months]
            filtered_data = filtered_data[filtered_data['Date'].dt.to_period('M').astype(str).isin(months)]

        if selected_customers:
            filtered_data = filtered_data[filtered_data['Customer Name'].isin(selected_customers)]

        if selected_items:
            filtered_data = filtered_data[filtered_data['Item Code'].isin(selected_items)]

        # Create a pivot table with Customer Name as rows and Item Code as columns
        pivot_table = pd.pivot_table(
            filtered_data,
            index='Customer Name',
            columns='Item Code',
            values='Quantity',
            aggfunc='count',  # Count the number of occurrences
            fill_value=0  # Fill missing values with 0
        )
        
        return pivot_table

    except Exception as e:
        st.error(f"Error generating the customer-item table: {e}")
        return pd.DataFrame()



# Function to load and process data
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    data['Complaint Date'] = pd.to_datetime(data['Complaint Date'])
    data['YearMonth'] = data['Complaint Date'].dt.to_period('M')
    return data

# Function to extract unique Fiscal Years from data in format YYYY-YYYY
def extract_fiscal_years(data):
    fiscal_years = data['Complaint Date'].dt.to_period('Q-APR').dt.year.unique()
    fiscal_years.sort()  # Sort in ascending order
    fiscal_years = [f"{year}-{year + 1}" for year in fiscal_years]
    fiscal_years.reverse()  # Reverse to get descending order
    return fiscal_years

def fetch_ytd_data(data, fiscal_year, year_offset):
    current_year = int(fiscal_year.split('-')[0])
    year = current_year - year_offset
    start_date_ytd = pd.Timestamp(f'{year}-04-01')
    end_date_ytd = pd.Timestamp(f'{year}-{pd.Timestamp.today().month:02d}-{pd.Timestamp.today().day:02d}')

    data_ytd = data[(data['Complaint Date'] >= start_date_ytd) & (data['Complaint Date'] <= end_date_ytd)]

    line_data_ytd = data_ytd[data_ytd['Priority'] == 'Line'].groupby(['Customer Group', 'YearMonth']).size().unstack(fill_value=0)
    warranty_data_ytd = data_ytd[data_ytd['Priority'] == 'Warranty'].groupby(['Customer Group', 'YearMonth']).size().unstack(fill_value=0)
    application_data_ytd=data_ytd[data_ytd['Priority'] == 'Application'].groupby(['Customer Group', 'YearMonth']).size().unstack(fill_value=0)
    
    line_data_ytd['FY'] = line_data_ytd.sum(axis=1)
    warranty_data_ytd['FY'] = warranty_data_ytd.sum(axis=1)
    application_data_ytd['FY'] = application_data_ytd.sum(axis=1)

    ytd_combined_data = pd.concat([line_data_ytd, warranty_data_ytd,application_data_ytd], axis=1, keys=['Line', 'Warranty','Application'])
    
    formatted_ytd_columns = [f'{col[0]}_{col[1].strftime("%b-%y")}' if isinstance(col[1], pd.Period) else f'{col[0]}_{col[1]}' for col in ytd_combined_data.columns]
    ytd_combined_data.columns = formatted_ytd_columns
    ytd_combined_data = ytd_combined_data.fillna(0)
    ytd_combined_data['Total'] = ytd_combined_data.sum(axis=1)
    return ytd_combined_data

def generate_combined_data(data, fiscal_year, selected_columns, display_fy_data, key_customers, selected_customer_groups, specific_columns, show_row_totals):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    line_data = data_current_fy[data_current_fy['Priority'] == 'Line'].groupby(['Customer Group', 'YearMonth']).size().unstack(fill_value=0)
    warranty_data = data_current_fy[data_current_fy['Priority'] == 'Warranty'].groupby(['Customer Group', 'YearMonth']).size().unstack(fill_value=0)
    application_data = data_current_fy[data_current_fy['Priority'] == 'Application'].groupby(['Customer Group', 'YearMonth']).size().unstack(fill_value=0)

    # Calculate FY totals
    line_data['FY'] = line_data.sum(axis=1)
    warranty_data['FY'] = warranty_data.sum(axis=1)
    application_data['FY'] = application_data.sum(axis=1)

    # Combine line and warranty data with month names as sub-columns
    combined_data = pd.concat([line_data, warranty_data, application_data], axis=1, keys=['Line', 'Warranty','Application'])

    # Flatten the multi-level column index and format the column names correctly
    formatted_columns = []
    for col in combined_data.columns:
        if isinstance(col, tuple):
            formatted_col = f'{col[0]}_{col[1].strftime("%b-%y")}' if isinstance(col[1], pd.Period) else f'{col[0]}_{col[1]}'
        else:
            formatted_col = col
        formatted_columns.append(formatted_col)

    combined_data.columns = formatted_columns

    # Apply key customers filter if selected
    if key_customers:
        key_customer_groups = ['KOEL', 'EICHER', 'M&M', 'AL', 'CAT', 'STERLING', 'OTHER']
        selected_customer_groups = list(set(selected_customer_groups + key_customer_groups))  # Combine with key customers

    # Ensure all selected customer groups are included without index errors
    all_customer_groups = data['Customer Group'].unique()
    selected_customer_groups = [group for group in selected_customer_groups if group in all_customer_groups]

    # Check if selected customer groups exist in the data
    not_found_customers = [customer for customer in selected_customer_groups if customer not in combined_data.index]
    if not_found_customers:
        st.warning(f"The following customer(s) were not found in the data: {', '.join(not_found_customers)}")
        st.stop()

    # Calculate the combined data again if there are new selected customer groups
    if selected_customer_groups != combined_data.index.tolist():
        new_groups = [group for group in selected_customer_groups if group not in combined_data.index]
        for group in new_groups:
            line_group = data_current_fy[(data_current_fy['Priority'] == 'Line') & (data_current_fy['Customer Group'] == group)].groupby('YearMonth').size()
            warranty_group = data_current_fy[(data_current_fy['Priority'] == 'Warranty') & (data_current_fy['Customer Group'] == group)].groupby('YearMonth').size()
            application_group = data_current_fy[(data_current_fy['Priority'] == 'Application') & (data_current_fy['Customer Group'] == group)].groupby('YearMonth').size()
            
            new_data = pd.concat([line_group, warranty_group, application_group], axis=1, keys=['Line', 'Warranty','Application']).fillna(0).sum(axis=1)
            combined_data.loc[group] = new_data

    # Add YTD data
    ytd_data = pd.DataFrame()
    for year_offset in [1]:  # Previous 1 year
        year = start_year - year_offset
        ytd_combined_data = fetch_ytd_data(data, fiscal_year, year_offset)

        if not ytd_combined_data.empty:
            if all(col in ytd_combined_data.columns for col in ['Line_FY', 'Warranty_FY','Application_FY']):
                ytd_combined_data = ytd_combined_data[['Line_FY', 'Warranty_FY','Application_FY']]
                ytd_combined_data.columns = ['Prev Year-Line_YTD', 'Prev Year-Warranty_YTD','Prev Year-Application_YTD']
                combined_data = pd.concat([combined_data, ytd_combined_data], axis=1, ignore_index=False)

                # Update column names after concatenation
                combined_data.columns = combined_data.columns.tolist()
            else:
                st.warning("YTD data does not contain 'Line_FY' or 'Warranty_FY' or 'Application_FY' columns.")

    # Ensure the order of customer groups is maintained
    customer_order = ['KOEL', 'EICHER', 'M&M', 'AL', 'CAT', 'STERLING', 'OTHER']
    ordered_selected_groups = [group for group in customer_order if group in selected_customer_groups]
    remaining_groups = [group for group in selected_customer_groups if group not in ordered_selected_groups]
    combined_data = combined_data.reindex(ordered_selected_groups + remaining_groups)

    # Replace NaN (NONE) with 0
    combined_data = combined_data.fillna(0)

    # Filter based on specific columns selected by the user
    if specific_columns:
        # Ensure specific columns exist in the DataFrame
        available_columns = [col for col in specific_columns if col in combined_data.columns]
        if available_columns:
            combined_data = combined_data[available_columns]
        else:
            st.warning("None of the specified columns in 'specific_columns' are available in the combined data.")
            combined_data = combined_data  # Keep the existing columns if none match

    if selected_columns:
        # Create a set of all possible columns based on selected_columns
        #all_selected_columns = [col for col in combined_data.columns if any(col.startswith(t) for t in selected_columns)]
        all_selected_columns = [col for col in combined_data.columns if any(t in col for t in selected_columns)]  #to handle prev-year line/warranty/application - not startswith

        if all_selected_columns:
            combined_data = combined_data[all_selected_columns]
        else:
            st.warning("None of the selected columns are available in the combined data.")
            combined_data = combined_data  # Keep the existing columns if none match


    # Calculate the total for each selected column and add it as a new row
    combined_data.loc['Total'] = combined_data.sum(axis=0)

    # Calculate the row total and add it as a new column if the button is clicked
    if show_row_totals:
        combined_data['Row_Total'] = combined_data.sum(axis=1)

    return combined_data


# Function to generate Combined Line and Warranty Complaints table
def generate_combined_complaints_table(data, fiscal_year, selected_months, show_line, show_warranty):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year + 1}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    if selected_months:
            data_current_fy['Month'] = data_current_fy['Complaint Date'].dt.strftime('%b-%y')
            data_current_fy = data_current_fy[data_current_fy['Month'].isin(selected_months)]


    combined_data = pd.DataFrame()

    if show_line:
        line_data = data_current_fy[data_current_fy['Priority'] == 'Line'].groupby(['Failure Code', 'Customer Group']).size().unstack(fill_value=0)
        combined_data = combined_data.add(line_data, fill_value=0)

    if show_warranty:
        warranty_data = data_current_fy[data_current_fy['Priority'] == 'Warranty'].groupby(['Failure Code', 'Customer Group']).size().unstack(fill_value=0)
        combined_data = combined_data.add(warranty_data, fill_value=0)

    if(show_line == False and show_warranty == False):
        st.warning("Select either Line or Warranty to view data")
        return 

    # Calculate row totals
    combined_data.loc['Total'] = combined_data.sum()
    combined_data['Total'] = combined_data.sum(axis=1)

    return combined_data

#plot for complaints page

# def generate_stacked_bar_chart_complaints(data, fiscal_year, show_line=True, show_warranty=True):
#     start_year = int(fiscal_year.split('-')[0])
#     end_year = int(fiscal_year.split('-')[1])

#     start_date = pd.Timestamp(f'{start_year}-04-01')
#     end_date = pd.Timestamp(f'{end_year}-03-31')

#     data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

#     combined_data = pd.DataFrame()

#     if show_line:
#         line_data = data_current_fy[data_current_fy['Priority'] == 'Line'].groupby(['Failure Code', 'Customer Group']).size().unstack(fill_value=0)
#         combined_data = combined_data.add(line_data, fill_value=0)

#     if show_warranty:
#         warranty_data = data_current_fy[data_current_fy['Priority'] == 'Warranty'].groupby(['Failure Code', 'Customer Group']).size().unstack(fill_value=0)
#         combined_data = combined_data.add(warranty_data, fill_value=0)

#     # Calculate row totals
#     combined_data.loc['Total'] = combined_data.sum()
#     combined_data['Total'] = combined_data.sum(axis=1)

#     # Remove the 'Total' row and column for plotting
#     data_to_plot = combined_data.drop('Total').drop(columns='Total')

#     # Melt the DataFrame to long format for Plotly
#     data_to_plot = data_to_plot.reset_index().melt(id_vars='Failure Code', var_name='Customer Group', value_name='Count')

#     # Create the stacked bar chart
#     fig = px.bar(data_to_plot, x='Failure Code', y='Count', color='Customer Group', title=f'Complaints per Failure Code by Customer Group for FY {fiscal_year}', barmode='stack')

#     # Display the chart in Streamlit
#     st.plotly_chart(fig)

#Function to plot trend for selected customer
def plot_trend(data, customer, fiscal_year):
    start_year = int(fiscal_year.split('-')[0])
    end_year = int(fiscal_year.split('-')[1])

    start_date = pd.Timestamp(f'{start_year}-04-01')
    end_date = pd.Timestamp(f'{end_year}-03-31')

    data_current_fy = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]

    line_data = data_current_fy[(data_current_fy['Priority'] == 'Line') & (data_current_fy['Customer Group'] == customer)].groupby(['YearMonth']).size()
    warranty_data = data_current_fy[(data_current_fy['Priority'] == 'Warranty') & (data_current_fy['Customer Group'] == customer)].groupby(['YearMonth']).size()
    application_data = data_current_fy[(data_current_fy['Priority'] == 'Application') & (data_current_fy['Customer Group'] == customer)].groupby(['YearMonth']).size()

    # Generate all months within the fiscal year
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')

    # Create traces
    fig = go.Figure()

    # Line Complaints
    fig.add_trace(go.Scatter(
        x=all_months.to_timestamp(),
        y=line_data.reindex(all_months, fill_value=0).values,
        mode='lines+markers+text',
        text=line_data.reindex(all_months, fill_value=0).values,
        textposition='top right',
        name='Line Complaints',
        line=dict(color='darkblue')
    ))

    # Warranty Complaints
    fig.add_trace(go.Bar(
        x=all_months.to_timestamp(),
        y=warranty_data.reindex(all_months, fill_value=0).values,
        name='Warranty Complaints',
        marker_color='#68ba9b'  # Set bar color
    ))

    # Application Complaints
    fig.add_trace(go.Scatter(
        x=all_months.to_timestamp(),
        y=application_data.reindex(all_months, fill_value=0).values,
        mode='lines+markers+text',
        text=application_data.reindex(all_months, fill_value=0).values,
        textposition='top right',
        name='Application Complaints',
        line=dict(color='orange')
    ))

    # Update layout
    fig.update_layout(
        title=f"Trend of Complaints for {customer} in {fiscal_year}",
        xaxis_title='Month',
        yaxis_title='Number of Complaints',
        xaxis=dict(type='category', tickvals=all_months.to_timestamp(), ticktext=all_months.strftime('%b')),
        yaxis=dict(title='Number of Complaints'),
        legend_title='Priority',
        barmode='overlay'
    )

    st.plotly_chart(fig)

    
def rgb_to_hex(rgb):
    return mcolors.to_hex(rgb)



def plot_ytd_fy_data(data, fiscal_year, year_offsets, line_checkbox, warranty_checkbox, application_checkbox, selected_customers):
    if not isinstance(year_offsets, list):
        year_offsets = [year_offsets]

    # Define color and line styles
    colors = plt.get_cmap('tab10').colors
    hex_colors = [rgb_to_hex(color) for color in colors]  # Convert colors to hex strings
    styles = ['solid', 'dash', 'dot', 'dashdot']

    # Initialize lists to store dataframes
    line_fy_data_frames = []
    warranty_fy_data_frames = []
    application_fy_data_frames = []

    for i, year_offset in enumerate(year_offsets):
        ytd_data = fetch_ytd_data(data, fiscal_year, year_offset)

        if 'Line_FY' not in ytd_data.columns and line_checkbox:
            st.warning(f"No 'Line_FY' data available to plot for year offset {year_offset}.")
        if 'Warranty_FY' not in ytd_data.columns and warranty_checkbox:
            st.warning(f"No 'Warranty_FY' data available to plot for year offset {year_offset}.")
        if 'Application_FY' not in ytd_data.columns and application_checkbox:
            st.warning(f"No 'Application_FY' data available to plot for year offset {year_offset}.")

        # Apply filters
        if selected_customers:
            ytd_data = ytd_data.loc[selected_customers]

        # Extract and fill data
        if line_checkbox:
            line_fy_data = ytd_data[['Line_FY']].copy().fillna(0)
            line_fy_data_frames.append(line_fy_data)
        if warranty_checkbox:
            warranty_fy_data = ytd_data[['Warranty_FY']].copy().fillna(0)
            warranty_fy_data_frames.append(warranty_fy_data)
        if application_checkbox:
            application_fy_data = ytd_data[['Application_FY']].copy().fillna(0)
            application_fy_data_frames.append(application_fy_data)

    # Combine all dataframes
    if line_checkbox:
        combined_line_fy_data = pd.concat(line_fy_data_frames, axis=1, keys=[f"Year {year_offset} Ago" for year_offset in year_offsets])
    if warranty_checkbox:
        combined_warranty_fy_data = pd.concat(warranty_fy_data_frames, axis=1, keys=[f"Year {year_offset} Ago" for year_offset in year_offsets])
    if application_checkbox:
        combined_application_fy_data = pd.concat(application_fy_data_frames, axis=1, keys=[f"Year {year_offset} Ago" for year_offset in year_offsets])

    # List of specific companies to include
    default_companies = selected_customers
    all_customers = data['Customer Group'].unique()
    if line_checkbox:
        combined_line_fy_data = combined_line_fy_data.reindex(all_customers, fill_value=0)
        combined_line_fy_data = combined_line_fy_data.loc[combined_line_fy_data.index.isin(default_companies)]
        combined_line_fy_data.index = combined_line_fy_data.index.astype(str)
    if warranty_checkbox:
        combined_warranty_fy_data = combined_warranty_fy_data.reindex(all_customers, fill_value=0)
        combined_warranty_fy_data = combined_warranty_fy_data.loc[combined_warranty_fy_data.index.isin(default_companies)]
        combined_warranty_fy_data.index = combined_warranty_fy_data.index.astype(str)
    if application_checkbox:
        combined_application_fy_data = combined_application_fy_data.reindex(all_customers, fill_value=0)
        combined_application_fy_data = combined_application_fy_data.loc[combined_application_fy_data.index.isin(default_companies)]
        combined_application_fy_data.index = combined_application_fy_data.index.astype(str)

    # Create Line_FY figure
    if line_checkbox:
        fig_line_fy = go.Figure()

        for i, year_offset in enumerate(year_offsets):
            line_fy_data = combined_line_fy_data.loc[:, (f"Year {year_offset} Ago", 'Line_FY')]
            fig_line_fy.add_trace(
                go.Scatter(
                    x=combined_line_fy_data.index,
                    y=line_fy_data,
                    mode='lines+markers+text',
                    name=f"Line FY Year {year_offset} Ago",
                    line=dict(color=hex_colors[i % len(hex_colors)], width=2, dash=styles[i % len(styles)]),
                    marker=dict(size=8),
                    text=[f'{v:.2f}' for v in line_fy_data],
                    textposition='top center'
                )
            )

        fig_line_fy.update_layout(
            title=f"Line FY Data for Fiscal Year {fiscal_year}",
            xaxis_title='Customer Group',
            yaxis_title='Line FY',
            xaxis_tickangle=-45,
            legend_title="Year Offset",
            grid=dict(rows=1, columns=1)
        )

        st.plotly_chart(fig_line_fy)

    # Create Warranty_FY figure
    if warranty_checkbox:
        fig_warranty_fy = go.Figure()

        for i, year_offset in enumerate(year_offsets):
            warranty_fy_data = combined_warranty_fy_data.loc[:, (f"Year {year_offset} Ago", 'Warranty_FY')]
            fig_warranty_fy.add_trace(
                go.Scatter(
                    x=combined_warranty_fy_data.index,
                    y=warranty_fy_data,
                    mode='lines+markers+text',
                    name=f"Warranty FY Year {year_offset} Ago",
                    line=dict(color=hex_colors[i % len(hex_colors)], width=2, dash=styles[i % len(styles)]),
                    marker=dict(size=8),
                    text=[f'{v:.2f}' for v in warranty_fy_data],
                    textposition='top center'
                )
            )

        fig_warranty_fy.update_layout(
            title=f"Warranty FY Data for Fiscal Year {fiscal_year}",
            xaxis_title='Customer Group',
            yaxis_title='Warranty FY',
            xaxis_tickangle=-45,
            legend_title="Year Offset",
            grid=dict(rows=1, columns=1)
        )

        st.plotly_chart(fig_warranty_fy)

    # Create Application_FY figure
    if application_checkbox:
        fig_application_fy = go.Figure()

        for i, year_offset in enumerate(year_offsets):
            application_fy_data = combined_application_fy_data.loc[:, (f"Year {year_offset} Ago", 'Application_FY')]
            fig_application_fy.add_trace(
                go.Scatter(
                    x=combined_application_fy_data.index,
                    y=application_fy_data,
                    mode='lines+markers+text',
                    name=f"Application FY Year {year_offset} Ago",
                    line=dict(color=hex_colors[i % len(hex_colors)], width=2, dash=styles[i % len(styles)]),
                    marker=dict(size=8),
                    text=[f'{v:.2f}' for v in application_fy_data],
                    textposition='top center'
                )
            )

        fig_application_fy.update_layout(
            title=f"Application FY Data for Fiscal Year {fiscal_year}",
            xaxis_title='Customer Group',
            yaxis_title='Application FY',
            xaxis_tickangle=-45,
            legend_title="Year Offset",
            grid=dict(rows=1, columns=1)
        )

        st.plotly_chart(fig_application_fy)



def main():
    # Set page configuration
    st.set_page_config(
        page_title="Warranty Service Data Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for additional styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f9f9f9;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
        }
        .stSelectbox, .stMultiselect, .stCheckbox {
            margin-top: 10px;
        }
        /* Custom styling for checkboxes and multiselect boxes */
        .stCheckbox>div>label>div {
            background-color: #d4edda; /* Light green background */
            border: 1px solid #28a745; /* Green border */
        }
        .stCheckbox>div>label>div input[type="checkbox"] {
            accent-color: #28a745; /* Green checkbox color */
        }
        .stMultiselect>div>label>div {
            background-color: #d4edda; /* Light green background */
            border: 1px solid #28a745; /* Green border */
        }
        .stMultiselect>div>label>div select {
            background-color: #d4edda; /* Light green background */
            color: #333; /* Dark text for contrast */
        }
        .stMultiselect>div>label>div select option {
            background-color: #d4edda; /* Light green background for options */
        }
        .stMultiselect>div>label>div select option:checked {
            background-color: #28a745; /* Dark green background for selected options */
            color: white; /* White text for selected options */
        }
        .stDataFrame {
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📊 Warranty Service Data Dashboard")

    # Upload the Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        data = map_frame_codes(data)

        # Extract all available Fiscal Years from the dataset
        fiscal_years = extract_fiscal_years(data)
        st.sidebar.image('NidecPower.png', width=230)

        # Sidebar table selection
        selected_table = st.sidebar.selectbox("Select Table", ["Cumulative Data", "Complaints", 
                                                               "Failure Code Analysis", "MTTR-Resolution Analysis",
                                                               "Zonal & Customer Analysis","FOC Register"])

        # Display the selected table based on user's choice
        if selected_table == "Cumulative Data":
            st.subheader("Cumulative Data")
            selected_fy = st.sidebar.selectbox("Select Fiscal Year", fiscal_years)

            # Checkbox widget for choosing data type (Line or Warranty) with default values set to True
            line_checkbox = st.sidebar.checkbox("Line", value=True)
            warranty_checkbox = st.sidebar.checkbox("Warranty", value=True)
            application_checkbox = st.sidebar.checkbox("Application", value=True)

            # Generate the Combined Data table based on filters
            selected_columns = []
            if line_checkbox:
                selected_columns.append("Line")
            if warranty_checkbox:
                selected_columns.append("Warranty")
            if application_checkbox:
                selected_columns.append("Application")

            key_customers = True  # Ensure key customers are always checked
            selected_customer_groups = st.sidebar.multiselect(
                "Select Customer Groups", 
                data['Customer Group'].unique(), 
                default=['KOEL', 'EICHER', 'M&M', 'AL', 'CAT', 'STERLING', 'OTHER']
            )

            selected_fy_index = fiscal_years.index(selected_fy)

            # Dummy combined data structure to get possible columns for selection
            dummy_fy = fiscal_years[selected_fy_index]
            dummy_combined_data = generate_combined_data(
                data, dummy_fy, selected_columns, False, key_customers, selected_customer_groups, None, False
            )

            # Add a button to toggle row totals
            show_row_totals = st.button("Show Row Totals")

            # Multi-select for specific columns
            specific_columns = st.sidebar.multiselect("Select Specific Columns", dummy_combined_data.columns)

            combined_data = generate_combined_data(
                data, selected_fy, selected_columns, False, key_customers, selected_customer_groups, specific_columns, show_row_totals
            )

            # Display the Combined Data table
            if combined_data.empty:
                st.warning("No data available based on current filters.")
            else:
                st.dataframe(combined_data,use_container_width=True)

            # Dropdown for selecting customer to show trend
            selected_customer = st.selectbox("Select Customer to View Trend", combined_data.index)
            plot_trend(data, selected_customer, selected_fy)

            # Add a button to show Year-to-Date (YTD) data
            if st.sidebar.button("Show YTD Data"):
                st.subheader("Year-to-Date (YTD) Data")
                ytd_data = fetch_ytd_data(data, selected_fy, 1)
                
                # Apply filters to YTD data
                if not ytd_data.empty:
                    # Filter by selected customer groups
                    ytd_data = ytd_data[ytd_data.index.isin(selected_customer_groups)]
                    
                    # Filter by selected columns
                    ytd_columns = [col for col in ytd_data.columns if any(col.startswith(t) for t in selected_columns)]
                    ytd_data = ytd_data[ytd_columns]

                    st.subheader("YTD Data for Previous Fiscal Year")
                    st.dataframe(ytd_data)
                
                ytd_data = fetch_ytd_data(data, selected_fy, 2)
                
                # Apply filters to YTD data
                if not ytd_data.empty:
                    # Filter by selected customer groups
                    ytd_data = ytd_data[ytd_data.index.isin(selected_customer_groups)]
                    
                    # Filter by selected columns
                    ytd_columns = [col for col in ytd_data.columns if any(col.startswith(t) for t in selected_columns)]
                    ytd_data = ytd_data[ytd_columns]

                    st.subheader("YTD Data for 2 Years Ago")
                    st.dataframe(ytd_data)

                # Plot Line FY data
                plot_ytd_fy_data(data, selected_fy, [0, 1, 2], line_checkbox, warranty_checkbox, application_checkbox, selected_customer_groups)

        elif selected_table == "Complaints":
            st.subheader("Line and Warranty Complaints")
            selected_fy = st.sidebar.selectbox("Select Fiscal Year", fiscal_years)

            # Checkbox for selecting complaint type
            show_line = st.sidebar.checkbox("Show Line", value=True)
            show_warranty = st.sidebar.checkbox("Show Warranty", value=True)

            start_year = int(selected_fy.split('-')[0])
            end_year = int(selected_fy.split('-')[1])
                
            start_date = pd.Timestamp(f'{start_year}-04-01')
            end_date = pd.Timestamp(f'{end_year}-03-31')

            months = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]['Complaint Date']
            months = months.dt.strftime('%b-%y').unique()
            selected_months = st.sidebar.multiselect("Select Months", months, default=[])


            # Generate and display Line or Warranty Complaints table based on filter
            complaints_table = generate_combined_complaints_table(data, selected_fy, selected_months, show_line, show_warranty)
            st.dataframe(complaints_table)

        elif selected_table == "Failure Code Analysis":
            st.subheader("Failure Code Analysis")
            selected_fy = st.sidebar.selectbox("Select Fiscal Year", fiscal_years)
            
            # Dropdown to select attribute for analysis
            attribute = st.sidebar.selectbox("Select Attribute", ["Rating", "Frame", "Location", "Hours Run"])

            # Generate default data for multiselect options
            if attribute == "Location":
                options = data['Location'].unique()
            elif attribute == "Frame":
                options = data['Frame'].unique()
            elif attribute == "Rating":
                options = data['Rating'].unique()
            elif attribute == "Hours Run":
                options = ['<50', '<100', '<200', '<500', '<750', '<1000', '<3000', '>3000']

            # Display multiselect for attribute values
            selected_values = st.sidebar.multiselect(
                f"Select {attribute} Values",
                options,
                default=[]  # Default to showing all options, but none are selected
            )

            start_year = int(selected_fy.split('-')[0])
            end_year = int(selected_fy.split('-')[1])
                
            start_date = pd.Timestamp(f'{start_year}-04-01')
            end_date = pd.Timestamp(f'{end_year}-03-31')

            months = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]['Complaint Date']
            months = months.dt.strftime('%b-%y').unique()
            selected_months = st.sidebar.multiselect("Select Months", months, default=[])

            selected_customer_groups = st.sidebar.multiselect(
                "Select Customer Groups", 
                data['Customer Group'].unique(), 
                default=[]
            )

            # Filter the dataset based on selected customer groups to get available OEMs
            filtered_data = data[data['Customer Group'].isin(selected_customer_groups)]

            # Add multi-select for OEMs
            selected_oems = st.sidebar.multiselect(
                "Select OEMs/Supplier", 
                filtered_data['Supplier/OEM'].unique(), 
                default=[]
            )

            # Generate and plot failure code analysis
            failure_code_data = generate_failure_code_analysis(data, selected_fy, attribute, selected_customer_groups, selected_oems, selected_months)
        
            # Display the table
            st.dataframe(failure_code_data)

            plot_failure_code_analysis(data, selected_fy, attribute, selected_values, selected_customer_groups, selected_oems, selected_months)

        elif selected_table == "MTTR-Resolution Analysis":
            st.subheader("MTTR Resolution Analysis")
            selected_fy = st.sidebar.selectbox("Select Fiscal Year", fiscal_years)
            
            status_filter = st.sidebar.selectbox("Select Status Filter", ["All", "Open", "Released"])

            selected_months = None
            if status_filter == 'Released' or "All":
                # Extract and display months for the selected fiscal year
                start_year = int(selected_fy.split('-')[0])
                end_year = int(selected_fy.split('-')[1])
                
                start_date = pd.Timestamp(f'{start_year}-04-01')
                end_date = pd.Timestamp(f'{end_year}-03-31')

                months = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]['Complaint Date']
                months = months.dt.strftime('%b-%y').unique()
                selected_months = st.sidebar.multiselect("Select Months", months, default=[])
            
            # Generate the resolution analysis data
            resolution_analysis_data = generate_resolution_analysis(data, selected_fy, status_filter, selected_months)
            resolution_analysis_data_percent = calculate_percentage_table(resolution_analysis_data)

            # Display the resolution analysis table
            st.write("By No. of Days for closure")
            st.dataframe(resolution_analysis_data)
            st.write("By Closure Percentages")
            st.dataframe(resolution_analysis_data_percent)

            # Plot the resolution analysis
            plot_resolution_analysis(resolution_analysis_data, status_filter)

        elif selected_table == "Zonal & Customer Analysis":
            st.subheader("Customer Group Analysis")
            selected_fy = st.sidebar.selectbox("Select Fiscal Year", fiscal_years)
            
            # Dropdown to select "Zonal In-Charge" name
            selected_name = st.sidebar.multiselect("Select Zonal In-Charge", data['Zonal In-Charge'].unique(), default=[])

            # Extract and display months
            start_year = int(selected_fy.split('-')[0])
            end_year = int(selected_fy.split('-')[1])
            
            start_date = pd.Timestamp(f'{start_year}-04-01')
            end_date = pd.Timestamp(f'{end_year}-03-31')

            months = data[(data['Complaint Date'] >= start_date) & (data['Complaint Date'] <= end_date)]['Complaint Date']
            months = months.dt.strftime('%b-%y').unique()
            selected_months = st.sidebar.multiselect("Select Months", months, default=[])

            
            # Generate the customer group analysis data
            customer_group_data = generate_customer_group_analysis(data, selected_fy, selected_name,selected_months)

            # Display the customer group analysis tables
            st.write("By Customer Group")
            st.dataframe(customer_group_data)

            plot_customer_group_analysis(customer_group_data)

        elif selected_table == "FOC Register":
            uploaded_file_foc = st.file_uploader("Choose your FOC File, let the above Warranty Complaints File Remain", type=["xlsx", "xlsm"], key="foc_file_uploader")
        
            if uploaded_file_foc is not None:
                df = pd.read_excel(uploaded_file_foc)

                if not df.empty:
                    # Extract fiscal years and filter data
                    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format
                    fiscal_years = extract_fiscal_years_foc(df)
                    selected_fy = st.sidebar.selectbox("Select Fiscal Year", fiscal_years)
                    
                    start_year = int(selected_fy.split('-')[0])
                    end_year = int(selected_fy.split('-')[1])
                    start_date = pd.Timestamp(f'{start_year}-04-01')
                    end_date = pd.Timestamp(f'{end_year}-03-31')

                    months = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]['Date']
                    months = months.dt.strftime('%b-%y').unique()
                    selected_months = st.sidebar.multiselect("Select Months", months, default=[])

                    # Select multiple customers and item codes
                    customers = df['Customer Name'].unique()
                    selected_customers = st.sidebar.multiselect("Select Customer Names", customers, default=[])

                    items = df['Item Code'].unique()
                    selected_items = st.sidebar.multiselect("Select Item Codes", items, default=[])

                    descriptions = df['Item Description'].unique()
                    selected_description = st.sidebar.selectbox("Select Item Description", descriptions)
                    
                    if selected_description:
                        item_code = get_item_code(df, selected_description)
                        if item_code:
                            st.sidebar.write(f"Item Code for '{selected_description}': {item_code}")

                    # Generate and display the customer-item table
                    customer_item_table = generate_customer_item_table(df, selected_fy, selected_months, selected_customers, selected_items)
                    if customer_item_table.empty:
                        st.warning("No data available based on current filters.")
                    else:
                        st.write("Customer-Item Count Table")
                        st.dataframe(customer_item_table)


# Run the app
if __name__ == "__main__":
    main()
