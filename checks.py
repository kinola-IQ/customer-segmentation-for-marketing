# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:48:03 2024

@author: akinola
"""

import pandas as pd
import charset_normalizer
import os
from typing import List, Optional, Union


def Musage(*dfs: pd.DataFrame) -> list:
    """
    Function to check memory usage information of DataFrame objects.

    Parameters:
    - *dfs: Variable length argument list of DataFrame objects to check memory usage for.

    Returns:
    - List of memory usage information for each DataFrame.
    """
    infos = []
    for df in dfs:
        try:
            infos.append(df.info(memory_usage='deep')) 
        except Exception as e:
            infos.append(f"Error processing DataFrame: {e}")
    return infos

def Null_Check(*dfs: pd.DataFrame) -> str:
    """
    Function to check for null values of df information of DataFrame objects.

    Parameters:
    - *dfs: Variable length argument list of DataFrame objects to check memory usage for.

    Returns:
    - String representation of null values for each DataFrame.
    """
    null_info = []
    for df in dfs:
        try:
            null_info.append(f"\n{df.isna().sum()}\n")
        except Exception as e:
            null_info.append(f"Error processing DataFrame: {e}\n")
    return ''.join(null_info)

def Null_by_dtype(*dfs: pd.DataFrame, Type: Optional[str] = None) -> str:
    """
    Function to check for null values in DataFrame objects by specified dtype.

    Parameters:
    - *dfs: Variable length argument list of DataFrame objects to check for null values.
    - Type: Optional dtype to filter columns for null check.

    Returns:
    - String representation of null values by dtype for each DataFrame.
    """
    null_info = []
    for df in dfs:
        try:
            null_info.append(f"\nMissing values by {Type}:\n{df.select_dtypes(include=[Type]).isna().sum()}\n")
        except Exception as e:
            null_info.append(f"Error processing DataFrame: {e}\n")
    return ''.join(null_info)

def Dup_Tot(*dfs: pd.DataFrame) -> List[int]:
    """
    Function to check the total number of duplicated rows in DataFrame objects.

    Parameters:
    - *dfs: Variable length argument list of DataFrame objects to check for duplicated rows.

    Returns:
    - List of total duplicated rows for each DataFrame.
    """
    duplicated_info = []
    for df in dfs:
        try:
            duplicated_info.append(df.duplicated().sum())
        except Exception as e:
            duplicated_info.append(f"Error processing DataFrame: {e}")
    return duplicated_info

def columns(*dfs: pd.DataFrame, dtype: Optional[Union[str, List[str]]] = None) -> str:
    """
    Function to get columns of specified dtype in DataFrame objects.

    Parameters:
    - *dfs: Variable length argument list of DataFrame objects to get columns for.
    - dtype: Optional dtype to filter columns.

    Returns:
    - String representation of columns by dtype for each DataFrame.
    """
    columns_info = []
    for df in dfs:
        try:
            columns_info.append(f"\n{df.select_dtypes(include=dtype).columns}\n")
        except Exception as e:
            columns_info.append(f"Error processing DataFrame: {e}\n")
    return ''.join(columns_info)

def process_date(dfs: List[pd.DataFrame], date_columns: List[str], fill_strategy: str = 'specific', specific_date: Optional[Union[str, pd.Timestamp]] = None) -> List[pd.DataFrame]:
    """
    Convert datetime columns to date format and fill missing values using a specified strategy.

    Parameters:
    - dfs (list of pd.DataFrame): List of DataFrames to process.
    - date_columns (list of str): List of date column names to convert and fill.
    - fill_strategy (str): Strategy to fill missing dates ('specific' or 'interpolate').
    - specific_date (str or datetime.date, optional): Specific date to fill missing values when fill_strategy is 'specific'.

    Returns:
    - list of pd.DataFrame: List of processed DataFrames.
    """
    for df in dfs:
        for col in date_columns:
            if col in df.columns:
                try:
                    # Convert to datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    if fill_strategy == 'specific':
                        # Fill missing values with a specific date
                        if specific_date is None:
                            specific_date = df[col].min().date()
                        df[col] = df[col].fillna(specific_date)
                    elif fill_strategy == 'interpolate':
                        # Interpolate missing values
                        df[col] = df[col].interpolate(method='time').fillna(df[col].min())
                    
                    # Extract just the date (year, month, day)
                    df[col] = df[col].dt.date
                except Exception as e:
                    print(f"Error processing column {col} in DataFrame: {e}")
    return dfs
def detect_encoding(file_path,sheet_name,column_To_check):
    """
    Detect the character encoding of the first few rows of a given sheet in an Excel file.

    Args:
    file_path (str): Path to the Excel file.
    sheet_name (str): Name of the sheet to read from.
    column_To_check(str): name of the column to check
    output_csv (str): Path to the CSV file to save the data.
    

    Returns:
    dict: Detected character encoding result.
    """
    # Load the Excel file
    raw_data = pd.ExcelFile(file_path)
    
    # Read the specified sheet
    sheet_name = pd.read_excel(raw_data, sheet_name)
    
    # Save the specified column to a CSV file
    default = sheet_name[column_To_check]
    default.to_csv(f"{column_To_check}.csv")
    
    # Open the CSV file and detect encoding
    with open(f"{column_To_check}.csv", 'rb') as rawdata:
        result = charset_normalizer.detect(rawdata.read(len(default)))
    
    # Print the result
    print(result)
    os.remove(f'{column_To_check}.csv')
    return result
