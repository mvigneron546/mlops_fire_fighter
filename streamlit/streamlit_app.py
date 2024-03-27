import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from helpers import (preprocess_date_columns,
                     preprocess_time_columns,
                     calculate_attendance_time)
from dotenv import load_dotenv

load_dotenv()

## YAML 
#import yaml
#yaml_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'config.yml'))
#with open(yaml_path, 'r') as f:
#    config = yaml.safe_load(f)

# ---------- Blob Storage Connection ----------

ACCOUNT_URL = 'https://firefighterdata.blob.core.windows.net'
CONTAINER_NAME = 'data'
DATA_PATH = 'newest_table.parquet'
OLD_DATA_PATH = 'oldies'

# Create the BlobServiceClient object
connect_str = os.getenv('STORAGE_ACCOUNT_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# ---------- Initiate Functions ----------

@st.cache_data 
def load_records(data_path):
  records = pd.read_parquet(BytesIO(
        container_client.download_blob(data_path).readall()))
  return records

@st.cache_resource
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    '''Applies the different preprocessing functions to the
    pandas DF.

    Args:
        - df: input dataframe

    Returns:
        A pd DataFrame
    '''
    df = preprocess_date_columns(df, 'DateOfCall')
    df = preprocess_time_columns(df, 'TimeOfCall')
    df = calculate_attendance_time(df)
#   df['NotionalCost'] = df['Notional Cost (£)']

#   df.drop(columns='Notional Cost (£)', inplace=True)
    df.dropna(subset=['FirstPumpArriving_AttendanceTime_min'], inplace=True)

    return df
  
# ---------- Streamlit Title ----------

st.title("ML-Ops Monitoring: :male-firefighter: :female-firefighter: :fire: :fire_engine: Firefighter Response Prediction")
st.sidebar.title("Table of contents")
pages=["Data Description", "Data Drift"]
page=st.sidebar.radio("Go to", pages)
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# https://github.com/scipy/scipy/issues/14298

# ---------- Import Dataesets ----------

currently_used_records = load_records(DATA_PATH)

current_date = pd.to_datetime(currently_used_records['DateOfCall'])
date_min = current_date.min().strftime('%Y-%m-%d')
date_max = current_date.max().strftime('%Y-%m-%d')
current_date = [f'{date_min}-{date_max}']

blob_list = container_client.list_blobs(name_starts_with=OLD_DATA_PATH)
blobfiles = []
for blob in blob_list:
    blobfiles.append(blob.name.split('/')[-1].split('_')[1].split('.')[0])


# ---------- Define Variables ----------

num_features = ['FirstPumpArriving_AttendanceTime_min', 
                'HourOfCall', 'NumStationsWithPumpsAttending',
                'NumPumpsAttending', 'NumCalls', 'DateOfCall_Month',
                ]

cat_features = ['IncidentGroup', 'StopCodeDescription', 'PropertyCategory',
                'PropertyType', 'AddressQualifier', 'ProperCase',
                'IncGeo_WardName', 'PartOfDay']

# ---------- First Page: Data Description ----------

if page == pages[0] : 
  st.header("Data Description")
  
  st.divider()
  st.caption('Target Variable')
  
  col1, col2 = st.columns(2)
  col1.markdown('***Name:*** FirstPumpArriving_AttendanceTime_min')
  col2.markdown('***Data Type:*** float64')
  st.markdown('***Description:*** First Pump attendance time in minutes')
  
  st.divider()
  st.caption('Numerical Variables')
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** HourOfCall')
  col2.markdown('**Data Type:** int64')
  st.markdown('**Description:** Hour of 999 call')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** NumStationsWithPumpsAttending')
  col2.markdown('**Data Type:** float64')
  st.markdown('**Description:** Number of stations with pumps in attendance')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** NumPumpsAttending')
  col2.markdown('**Data Type:** float64')
  st.markdown('**Description:** Number of pumps in attendance')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** NumCalls')
  col2.markdown('**Data Type:** float64')
  st.markdown('**Description:** Number of calls')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** DateOfCall_Month')
  col2.markdown('**Data Type:** int64')
  st.markdown('**Description:** Month of 999 call')
  st.markdown("") 
  
  st.divider()
  st.caption('Categorical Variables')
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** IncidentGroup')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** High level incident category')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** StopCodeDescription')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** Detailed incident category')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** PropertyCategory')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** High level property descriptor')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** PropertyType')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** Detailed property descriptor')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** AddressQualifier')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** Qualifies location of actual incident relevant to category above')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** ProperCase')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** Borough Name')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** IncGeo_WardName')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** Ward Name')
  st.markdown("") 
  
  col1, col2 = st.columns(2)
  col1.markdown('**Name:** PartOfDay')
  col2.markdown('**Data Type:** object')
  st.markdown('**Description:** Part of day of 999 call')

# ---------- Second Page ----------

if page == pages[1] : 
  st.header("Data Drift")
  
  st.write("Please select the London Fire Brigade Records you would like to compare:")
  col1, col2 = st.columns(2)
  option1 = col1.selectbox(
      f'Current Records',
      current_date)
  option2 = col2.selectbox(
      f'Previous Records',
      blobfiles)
  st.write(f"You are comparing the recrods from \"{option1}\" to \"{option2}\".")
  
  ## Load previous records
  previous_used_records = load_records(OLD_DATA_PATH + '/table_' + option2 + '.parquet')
  
  st.write(f"Please be patient for a moment while the data is processed.")
  
  ## Process records
  currently_used_records_processed = preprocess_df(currently_used_records)
  old_used_records_processed = preprocess_df(previous_used_records)
  
  currently_used_records_processed['Record'] = 'New'
  old_used_records_processed['Record'] = 'Previous'
  df = pd.concat([currently_used_records_processed, old_used_records_processed], ignore_index=True)
  
  st.divider()

  st.write("Please select a variable that you would like to compare:")
  variable = st.selectbox(
      'Select Variable',
      num_features + cat_features)

  st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
  )
  
  # ---------- Numerical Data ----------
  
  if variable in num_features: 
    col1, col2 = st.columns([3, 1])
    
    fig = plt.figure()
    sns.kdeplot(x=variable, data=df, hue='Record', multiple="layer")
    plt.title(f"Distribution of {variable}")
    col1.pyplot(fig)
    
    test_statistic = stats.ttest_ind(
    currently_used_records_processed[variable].dropna(), 
    old_used_records_processed[variable].dropna())
    if test_statistic.pvalue > 0.05:
      col2.metric("Two-Tailed T-Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='normal'
                )
    else: 
      col2.metric("Two-Tailed T-Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='inverse'
                )
      
    test_statistic = stats.kstest(
    currently_used_records_processed[variable].dropna(), 
    old_used_records_processed[variable].dropna())
    if test_statistic.pvalue > 0.05:
      col2.metric("Kolmogorov-Smirnov Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='normal'
                )
    else: 
      col2.metric("Kolmogorov-Smirnov Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='inverse'
                )
  
  # ---------- Categorical Data ----------
  
  elif variable in cat_features: 
    col1, col2 = st.columns([3, 1])
    
    fig = plt.figure()
    sns.countplot(x=variable, data=df, hue='Record')
    plt.xticks(rotation=40, ha='right')
    plt.title(f"Distribution of {variable}")
    col1.pyplot(fig)
    
    ct_table_ind=pd.crosstab(
      currently_used_records_processed[variable].dropna(),
      old_used_records_processed[variable].dropna())
    test_statistic =  stats.chi2_contingency(ct_table_ind)
    if test_statistic.pvalue > 0.05:
      col2.metric("Chi-Square Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='normal'
                )
    else: 
      col2.metric("Chi-Square Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='inverse'
                )
    
    test_statistic = stats.kstest(
    currently_used_records_processed[variable].dropna(), 
    old_used_records_processed[variable].dropna())
    if test_statistic.pvalue > 0.05:
      col2.metric("Kolmogorov-Smirnov Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='normal'
                )
    else: 
      col2.metric("Kolmogorov-Smirnov Test", 
                f"{test_statistic.statistic:.5g}",
                f"P-Value: {test_statistic.pvalue:.3g}",
                delta_color='inverse'
                )
    
  st.divider()