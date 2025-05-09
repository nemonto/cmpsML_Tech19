#%% MODULE BEGINS
module_name = '<feature_extraction>'
'''
Version: <5.0>
Description:
<Prepares data to be input into the ML models.>
Authors:
<Drew Hutchinson, Zichuo Wang>
Date Created : <4-13-2025>
Date Last Updated: <5-8-2025>
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
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def drop_missing_values(df):
    return df.dropna()

def drop_irrelevant_columns(df):
    return df.drop(columns=['year','SID'], errors='ignore')

def one_hot_encode(df):
    df = df.copy()
    df['sex'] = df['sex'].astype(str)
    df['island'] = df['island'].astype(str)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=['island'], drop_first=False, dtype=int)
    return df

def label_encode_species(df):
    df = df.copy()
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    return df

def preprocessing_penguins(df):
    df = df.copy()
    df = drop_missing_values(df)
    df = drop_irrelevant_columns(df)
    df = one_hot_encode(df)
    df = label_encode_species(df)
    return df

def split_data(df):
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    random_state = 42
    df = df.copy()
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=random_state)
    train,val = train_test_split(train_val, test_size=val_ratio/(train_ratio + val_ratio), random_state=random_state)

    return train,val,test

def noisify(df):
    noise_std_dict = {
        'bill_length_mm': 3.0,
        'bill_depth_mm': 1.5,
        'flipper_length_mm': 7.0,
        'body_mass_g': 150.0
    }

    df = df.copy()

    for col, std in noise_std_dict.items():
        if col in df.columns:
            noise = np.random.normal(loc=0, scale=std, size=df[col].shape)
            df[col] += noise
        else:
            print(f"Column '{col}' not found in the dataset.")

    return df

def standardize_data(df1,df2,df3):
    df1 = df1.copy()
    df2 = df2.copy()
    df3 = df3.copy()
    numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    scaler = StandardScaler()
    cols_to_scale = [col for col in numeric_cols if col in df1.columns]
    df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])
    df2[cols_to_scale] = scaler.transform(df2[cols_to_scale])
    df3[cols_to_scale] = scaler.transform(df3[cols_to_scale])
    return df1,df2,df3

def plot_distribution(df1,df2,df3,plots_dir):

    df1["split"] = "train"
    df2["split"] = "val"
    df3["split"] = "test"

    df_all = pd.concat([df1, df2, df3], axis=0)

    exclude_cols = ["species", "split"]
    numeric_cols = df_all.select_dtypes(include="number").columns.drop(exclude_cols, errors="ignore")

    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.kdeplot(data=df_all, x=col, hue="split", common_norm=False)
        plt.title(f"Distribution of '{col}' in Train / Val / Test")
        plt.tight_layout()
        plot_path = plots_dir / f'{col}_Train_Val_Test.png'
        plt.savefig(plot_path)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_all, x='species', hue='split')
    plt.title("Species Distribution in Train / Val / Test")
    plt.xlabel("Penguin Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plot_path = plots_dir / 'Species_Train_Val_Test.png'
    plt.savefig(plot_path)
    plt.show()

def plot_data(df, ax, title, x_col, y_col, plot_style):
    species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    df = df.copy()
    # reconstruct species,island and sex columns
    df['species_name'] = df['species'].map(species_map)
    df['island'] = df[['island_Biscoe', 'island_Dream', 'island_Torgersen']].idxmax(axis=1).str.replace('island_', '')
    df['sex'] = df['sex_male'].apply(lambda x: 'Male' if x == 1 else 'Female')
    # plot
    if plot_style == 1:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='species_name', style='species_name', ax=ax)
    elif plot_style == 2:
        sns.boxplot(data=df, x=x_col, y=y_col, hue='species_name', ax=ax)
    elif plot_style == 3:
        sns.violinplot(data=df, x=x_col, y=y_col, hue='species_name', ax=ax)

    ax.set_title(title)
    ax.legend(title='Species')

def save_data(df_dict, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for sheetname, df in df_dict.items():
            df.to_excel(writer, sheet_name= sheetname, index=False)

def main():
    #file path declaration
    base_dir = Path(__file__).resolve().parent.parent
    input_original_path = base_dir / "CODE" / "INPUT" / "penguins.csv"
    input_path = base_dir / "CODE" / "INPUT" / "feature_extracted_data.xlsx"
    input_train_path = base_dir / "CODE" / "INPUT" / "TRAIN" / "train.xlsx"
    input_test_path = base_dir / "CODE" / "INPUT" /"TEST"/ "val_test.xlsx"
    output_path = base_dir / "CODE" / "OUTPUT" / "feature_extracted_data.xlsx"
    plots_dir = base_dir / "CODE" / "OUTPUT" / "dataset plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    #read original file from input folder
    D0 = pd.read_csv(input_original_path)
    print("Original dataset D0:")
    print(f"This Data Set has {D0.shape[0]} rows, {D0.shape[1]} columns")
    print(D0.head())

    #data preprocessing - drop columns and data encoding
    D1 = preprocessing_penguins(D0)
    print("Preprocessed dataset D1:")
    print(f"This Data Set has {D1.shape[0]} rows, {D1.shape[1]} columns")
    print(D1.head())

    #dataset splitting
    train,val,test = split_data(D1)
    print("Training dataset:")
    print(f"This Dataset has {train.shape[0]} rows, {train.shape[1]} columns")
    print(train.head())
    print("Validation dataset:")
    print(f"This Dataset has {val.shape[0]} rows, {val.shape[1]} columns")
    print(val.head())
    print("Test dataset:")
    print(f"This Dataset has {test.shape[0]} rows, {test.shape[1]} columns")
    print(test.head())

    #add noise to the training dataset
    train_noise = noisify(train)

    #applying scaler to sub-sets
    train_std,val_std,test_std = standardize_data(train_noise,val,test)

    #output the datasets to excel file
    save_data({
        'Whole': D1,
        'Train': train,
        'Train_noise': train_noise,
        'Val': val,
        'Test': test,
        'Train_noise_std': train_std,
        'Val_std': val_std,
        'Test_std': test_std
    }, output_path)

    #output the datasets to INPUT, INPUT\TRAIN and INPUT|TEST folders for further using
    save_data({
        'Whole': D1,
        'Train': train,
        'Train_noise': train_noise,
        'Val': val,
        'Test': test,
        'Train_noise_std': train_std,
        'Val_std': val_std,
        'Test_std': test_std
    }, input_path)
    save_data({'Train_noise_std': train_std,'Train_noise': train_noise}, input_train_path)
    save_data({'Val_std': val_std,'Test_std': test_std, 'Val': val, 'Test': test}, input_test_path)

    #output the plots of train/val/test distribution
    plot_distribution(train_std,val_std,test_std,plots_dir)
    #output the plots of D1
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plot_data(D1, axes[0, 0], "D1: Scatter Plot of Bill Length vs Bill Width", 'bill_length_mm', 'bill_depth_mm', 1)
    plot_data(D1, axes[0, 1], "D1: Box Plot of Sex vs Body Mass", 'sex', 'body_mass_g', 2)
    plot_data(D1, axes[1, 0], "D1: Violin Plot of Islands vs Body Mass", 'island', 'body_mass_g', 3)
    axes[1, 1].remove()
    plt.suptitle("Feature Extraction Plots")
    plt.tight_layout(rect = [0, 0, 1, 0.96])
    plot_path = plots_dir / 'Tech19_PA2_Plots.png'
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
