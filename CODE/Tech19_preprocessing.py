#%% MODULE BEGINS
module_name = '<preprocessing>'
'''
Version: <4.0>
Description:
<Preprocesses data and generates plots.>
Authors:
<Drew Hutchinson, Zichuo Wang>
Date Created : <3-30-2025>
Date Last Updated: <5-6-2025>
Doc:
<***>
Notes:
<***>
'''
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
#os.chdir("./../..")
#
#custom imports
#other imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here

def handle_missing_values(df):
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].dtype == 'object':
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
        else:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    return df_imputed

def drop_missing_values(df):
    return df.dropna()

def normalize_data(df):
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    cols_to_normalize = df.columns[3:7]
    df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    return df_normalized

def standardize_data(df):
    df_standardized = df.copy()
    scaler = StandardScaler()
    cols_to_standardize = df.columns[3:7]
    df_standardized[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
    return df_standardized

def plot_data(df, ax, title, x_col, y_col):
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='species', style='species', ax=ax)
    ax.set_title(title)

def save_data(df_dict, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for sheetname, df in df_dict.items():
            df.to_excel(writer, sheet_name= sheetname, index=False)

def main():
    #file path declaration
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "CODE" / "INPUT" / "penguins.csv"
    output_path = base_dir / "CODE" / "OUTPUT" / "preprocessed_data.xlsx"
    plots_dir = base_dir / "CODE" / "OUTPUT" / "dataset plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    #read original file from input folder
    D0 = pd.read_csv(input_path)
    print("Original data set D0:")
    print(f"This Data Set has {D0.shape[0]} rows, {D0.shape[1]} columns")
    print(D0.head())

    #data preprocessing - generating different datasets
    D1 = handle_missing_values(D0)
    print("Imputed missing values data set D1:")
    print(f"This Data Set has {D1.shape[0]} rows, {D1.shape[1]} columns")
    print(D1.head())

    D2 = drop_missing_values(D0)
    print("Dropped missing values data set D2:")
    print(f"This Data Set has {D2.shape[0]} rows, {D2.shape[1]} columns")
    print(D2.head())

    D3 = normalize_data(D2)
    print("Normalized data set D3:")
    print(f"This Data Set has {D3.shape[0]} rows, {D3.shape[1]} columns")
    print(D3.head())

    D4 = standardize_data(D2)
    print("Standardized data set D4:")
    print(f"This Data Set has {D4.shape[0]} rows, {D4.shape[1]} columns")
    print(D4.head())

    #output the datasets to excel file
    save_data({'Original D0': D0, 'Imputed D1': D1, 'Dropped D2': D2, 'Normalized D3': D3, 'Standardized D4': D4}, output_path)

    #output the plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plot_data(D0, axes[0, 0], "D0: Bill Length vs Bill Width", 'bill_length_mm', 'bill_depth_mm')
    plot_data(D1, axes[0, 1], "D1: Imputed Bill Length vs Flipper Length", 'bill_length_mm', 'flipper_length_mm')
    plot_data(D2, axes[0, 2], "D2: Dropped Bill Length vs Body Mass", 'bill_length_mm', 'body_mass_g')
    plot_data(D2, axes[1, 0], "D2: Dropped Flipper Length vs Body Mass", 'flipper_length_mm', 'body_mass_g')
    plot_data(D3, axes[1, 1], "D3: Normalized Flipper Length vs Body Mass", 'flipper_length_mm', 'body_mass_g')
    plot_data(D4, axes[1, 2], "D4: Standardized Flipper Length vs Body Mass", 'flipper_length_mm', 'body_mass_g')
    plt.suptitle("Data Preprocessing Plots")
    plt.tight_layout(rect = [0, 0, 1, 0.96])
    plot_path = plots_dir / 'Tech19_PA1_Plots.png'
    plt.savefig(plot_path)
    plt.show()
    plt.close()




#
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here

#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
#TEST Code
main()
