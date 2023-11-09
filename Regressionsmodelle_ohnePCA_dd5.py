#%% Import libaries

# NumPy is the fundamental package for scientific computing with Python.
import numpy as np
# Matplotlib is a Python 2D plotting library
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

# BioSPPy is a toolbox for biosignal processing
import biosppy as bp

#pandas for reading the data
import pandas as pd

def read_xlsx_from_github(url, sheet_number):
    """
    Reads the first sheet of an XLSX file from a GitHub URL and returns it as a NumPy array.

    Args:
    url (str): The GitHub URL of the XLSX file.

    sheet_number(): page of excel sheet starting with 0

    Returns:
    numpy.ndarray: A NumPy array containing the data from the first sheet.
    """
    try:
        # Read the XLSX file into a DataFrame
        if sheet_number==2:
          data = pd.read_excel(url, sheet_name=sheet_number, header=None)
        else:
          data = pd.read_excel(url, sheet_name=sheet_number)
        # Convert the DataFrame to a NumPy array
        numpy_array = data.to_numpy()

        return numpy_array
    except Exception as e:
        print("An error occurred:", str(e))
        return None

# Getting xlsx Data:
github_link_test_data = "https://github.com/CremaschiLab/Cardiac_Differentiation_Modeling/raw/master/test_data_corrected.xlsx"
github_link_train_data = "https://github.com/CremaschiLab/Cardiac_Differentiation_Modeling/raw/master/train_data.xlsx"

x_test_data = read_xlsx_from_github(github_link_test_data, 0)
y_test_data = read_xlsx_from_github(github_link_test_data, 1)
labels_name_test_data = read_xlsx_from_github(github_link_test_data, 2)

x_train_data = read_xlsx_from_github(github_link_train_data, 0)
y_train_data = read_xlsx_from_github(github_link_train_data, 1)
labels_name_train_data = read_xlsx_from_github(github_link_train_data, 2)

features = np.vstack((x_test_data, x_train_data))
labels = np.vstack((y_test_data, y_train_data))
#print(merged_array.shape)
#print(labels_test_data.shape)
#print(features.shape)

feature_vector = pd.DataFrame(features, columns=labels_name_test_data)
# print("label.shape and label_name_test_data")
# print(labels.shape)
# print(labels_name_test_data.shape)
#feature_vector


#%% Scaling Features -> Data Matrix for dd5 and dd7
from sklearn.preprocessing import StandardScaler
features_std_dd7 = features

# Liste der Spaltennamen
column_names = labels_name_test_data.ravel()
column_names= column_names.astype(str)

# Finding indices, that have dd6, dd7 or Overall in their names, so we only have the data from the days before
# this can be extended to any differentiation day
indices_to_remove = [i for i, name in enumerate(column_names) if 'dd7' in name or 'Overall' in name or 'dd6' in name or 'dd5' in name or 'dd4' in name or 'dd3' in name]

# Entferne die entsprechenden Spalten aus der Matrix
X_dd5 = np.delete(features_std_dd7, indices_to_remove, axis=1)
label_names_dd5 = np.delete(column_names, indices_to_remove, axis=0)
X_dd7 = features_std_dd7

feature_vector_std_dd7 = pd.DataFrame(features_std_dd7, columns=labels_name_test_data)
feature_vector_std_dd5 = pd.DataFrame(X_dd5, columns=label_names_dd5)


# #%% Box-Plots of the Scaled Features
# features_per_plot = 17

# # Anzahl der Plots
# num_plots = X_dd7.shape[1] // features_per_plot

# # Größe der Figure
# fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))

# for i in range(num_plots):
#     # Start- und Endindex für die Features in diesem Plot
#     start_idx = i * features_per_plot
#     end_idx = (i + 1) * features_per_plot

#     # Features für diesen Plot auswählen
#     features = X_dd7[:, start_idx:end_idx]

#     # Boxplot erstellen
#     axes[i].boxplot(features, vert=False)

#     # Achsentitel setzen
#     axes[i].set_title(f'Features {start_idx+1}-{end_idx}')

# plt.tight_layout()
# plt.show()


#%% Train-Test-Split
from sklearn.model_selection import train_test_split

# X_train, y_train, y_test and so on are the variables used for training the model. The ending dd7 or dd5 indicates until which day the measurements/features are.
"""
To change the test size, change the variable "test_size":

"""
test_size = 0.15

X_train_dd5, X_test_dd5, y_train_dd5, y_test_dd5 = train_test_split(X_dd5, labels, test_size=test_size, random_state=42)

X_train_dd7, X_test_dd7, y_train_dd7, y_test_dd7 = train_test_split(X_dd7, labels, test_size=test_size, random_state=42)

#%% scaling
Scaler1 = StandardScaler().fit(X_train_dd7)
X_train_dd7 = Scaler1.transform(X_train_dd7)
X_test_dd7 = Scaler1.transform(X_test_dd7)

Scaler2 = StandardScaler().fit(X_train_dd5)
X_train_dd5 = Scaler2.transform(X_train_dd5)
X_test_dd5 = Scaler2.transform(X_test_dd5)

#%% implementation
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold  
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
from matplotlib import gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import seaborn as sns  # Stelle sicher, dass diese Zeile vorhanden ist
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, RationalQuadratic


# X_train: Train Data (54 Beobachtungen, 81 Merkmale)
# y_train: Train target (54 Beobachtungen, 1 target)
# X_test: Test Data (6 Beobachtungen, 81 Merkmale)
# y_test: Test target (6 Beobachtungen, 1 target)


#%% Define variables
mses_pcrpca = []  # Array für die Mean Squared Errors (MSE)
r_squares_pcrpca = []  # Array für die R-squared (R2)
mses_lassopca = []  # Array für die Mean Squared Errors (MSE)
r_squares_lassopca = []  # Array für die R-squared (R2)
variance=np.arange(0.8, 1.02, 0.02)  # range of variance from 60% to 100%
alpha = 0.9  # alpha for lasso regression
threshold_corr = 0.99  # threshold for removing correlated features
th = []   # Array for threshold to plot it in table
k = 9
n_features=60
n_features_to_select = np.arange(57, 59, 2)


#%% Lasso Regression

def lasso(X_train, X_test, y_train, y_test, selected_column_indices):
    # Lasso Regression
    lasso = LassoCV(cv=5 ,max_iter=10000)
    
    lasso.fit(X_train, y_train.ravel())
    print('alpha: ', lasso.alpha_)

    # Wichtigste Koeffizienten von Lasso
    lasso_coefs = lasso.coef_
    print('Koeffizienten: ', lasso_coefs)
    print('Shape of selected features: ', len(selected_column_indices))
    # print('selected_column_indices: ', selected_column_indices)

    weighted_column_indices = [index for index, coef in enumerate(lasso_coefs) if abs(coef) > 0]

    abs_coefs = [abs(lasso_coefs[i]) for i in weighted_column_indices]

    # Sort indices based on absolute coefficients
    sorted_indices = [index for _, index in sorted(zip(abs_coefs, weighted_column_indices), reverse=True)]


    # print('After LASSO: ', weighted_column_indices)
    # print('Shape of weighted features with LASSO: ', len(weighted_column_indices))    

    # print(column_names[weighted_column_indices])

    # #Balkendiagramm erstellen
    # fig, ax = plt.subplots(figsize=(15, 6))
    # ax.barh(column_names[sorted_indices], abs(lasso_coefs[sorted_indices]))
    # ax.set_xlabel('Coefficients', fontsize=15)
    # ax.set_ylabel('Features', fontsize=15, labelpad=50)
    # ax.set_title('12 highest weighted features by Lasso-regression for differentiation day 7', fontsize=18)
    # ax.tick_params(axis='both', labelsize=14)
    # ax.grid(axis='x')
    # plt.tight_layout()
    # plt.show()

    # Calculate R² on the test set
    r_squared = lasso.score(X_test, y_test)

    # Cross Validation
    scores = -cross_val_score(lasso, X_test, y_test.ravel(), cv=k, scoring='neg_root_mean_squared_error')
    
    # Predictions for test data
    y_pred = lasso.predict(X_test)

    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)

    # Calculate and print the mean and standard deviation of the scores
    mean = scores.mean()
    std = scores.std()

    # Residuen berechnen
    residuals = y_test - lasso.predict(X_test)

    print(residuals)
    print('lassopredict')
    print(lasso.predict(X_test))

    # Residualplot erstellen
    plt.figure(figsize=(8, 6))
    sns.residplot(x=lasso.predict(X_test).flatten(), y=residuals[:, 3].flatten(), lowess=True, scatter_kws={'alpha': 0.5})
    
# X-Achse genauer anpassen
    plt.xlim(min(lasso.predict(X_test).flatten())-5, max(lasso.predict(X_test).flatten())+5)

    plt.title('Residual Plot')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.show()
    
    return mean, std, r_squared



#%% Create Table for evaluation
def createtable(ax, name, var, value_mean, value_std, r_squared):
  # Erstellen eines DataFrames aus den Listen
  table1 = {'number of features': var,'RMSE': value_mean, 'STD': value_std}
  df1 = pd.DataFrame(table1)

  
  ax.axis('off')  # Deaktivieren der Achsen
  table_data1 = []
  table_data1.append(df1.columns.tolist())
  table_data1.extend(df1.values.tolist())


  # Erstellen der ersten Tabelle
  cell_text1 = ax.table(cellText=df1.values, colLabels=df1.columns, loc='center')
  cell_text1.auto_set_font_size(False)
  cell_text1.set_fontsize(12)
  #ax.set_title(name, fontsize=12)
  cell_text1[0, 0].set_facecolor('#C2BCE0')
  cell_text1[0, 1].set_facecolor('#C2BCE0')
  cell_text1[0, 2].set_facecolor('#C2BCE0')
  #cell_text1[0, 3].set_facecolor('#C2BCE0')


#%%
def evaluate(title, X_train, X_test, Y_train, Y_test):
  means_lasso_mi= []
  means_pcr_mi= []
  stds_lasso_mi= []
  stds_pcr_mi = []
  means_lasso_rfe= []
  means_pcr_rfe= []
  stds_lasso_rfe = []
  stds_pcr_rfe = []
  r_squares_lasso_rfe=[]
  r_squares_lasso_mi=[]
    
#   # Removing Correlated Features
#   # Berechnen der Korrelationsmatrix
#   correlation_matrix = np.corrcoef(X_train, rowvar=False)
#   # In rowvar=False, die Merkmale sind in den Spalten des Arrays

#   # Erzeugen einer booleschen Maske für die obere Dreiecksmatrix der Korrelationsmatrix
#   mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
#   # Finden von stark korrelierten Merkmalspaaren
#   highly_correlated = np.where((correlation_matrix > threshold_corr) & (correlation_matrix < 1.0))
#   # Entfernen der stark korrelierten Merkmale (falls gewünscht)
#   X_train_corr = np.delete(X_train, highly_correlated[0], axis=1)
#   X_test_corr = np.delete(X_test, highly_correlated[0], axis=1)
  for n_features in n_features_to_select :
# Create an RFE estimator with the specified number of features to select
    model = LinearRegression()
    estimator_rfe = RFE(model, n_features_to_select=n_features, step=1)
    estimator_rfe = estimator_rfe.fit(X_train, Y_train.ravel())
    
    # Get the selected features
    selected_columns = pd.DataFrame(X_train).columns[estimator_rfe.support_]
    X_train_rfe = X_train[:, estimator_rfe.support_]
    X_test_rfe = X_test[:, estimator_rfe.support_]

    # Holen Sie sich die ausgewählten Spaltennummern
    selected_column_indices_rfe = [i for i, support in enumerate(estimator_rfe.support_) if support]


# Perform feature selection using Mutual Information
    selector = SelectKBest(score_func=mutual_info_regression, k=n_features if n_features <= X_train.shape[1] else 'all')
    X_train_mi = selector.fit_transform(X_train, Y_train)
    X_test_mi = selector.transform(X_test) 

    # Holen Sie sich die ausgewählten Spaltennummern
    selected_column_indices_mi = [i for i, mask in enumerate(selector.get_support()) if mask]

    
    # PCR and Lasso with rfe
    mean_lasso, std_lasso, r_square = lasso(X_train_rfe, X_test_rfe, Y_train, Y_test, selected_column_indices_rfe)
    mean_lasso_round=int(round(mean_lasso))
    std_lasso_round=round(std_lasso)
    r_square_round = round(r_square, 2)

    means_lasso_rfe.append(mean_lasso_round)
    stds_lasso_rfe.append(std_lasso_round)
    r_squares_lasso_rfe.append(r_square_round)


    # PCR und LASSO with MI
    mean_lasso_mi, std_lasso_mi, r_square_mi = lasso(X_train_mi, X_test_mi, Y_train, Y_test, selected_column_indices_mi)
    mean_lasso_round_mi=int(round(mean_lasso_mi))
    std_lasso_round_mi=round(std_lasso_mi)
    r_square_round_mi = round(r_square_mi, 2)

    means_lasso_mi.append(mean_lasso_round_mi)
    stds_lasso_mi.append(std_lasso_round_mi)
    r_squares_lasso_mi.append(r_square_round_mi)  



  # Erstellen einer Tabelle im Plot
  fig, ax = plt.subplots(1, 2, figsize=(18, 9))
  # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 5])
  # ax = []
  # for i in range(2):
  #   for j in range(2):
  #       ax.append(plt.subplot(gs[i, j]))
  createtable(ax[0], 'RFE', n_features_to_select, means_lasso_rfe, stds_lasso_rfe, r_squares_lasso_rfe)
  createtable(ax[1], 'MI', n_features_to_select, means_lasso_mi, stds_lasso_mi, r_squares_lasso_mi)
  # createtable(ax[2], 'mean rfe', n_features_to_select, means_lasso_rfe, means_pcr_rfe)
  # createtable(ax[3], 'std rfe', n_features_to_select, stds_lasso_rfe, stds_pcr_rfe)
  fig.suptitle(title)
  plt.get_current_fig_manager().window.showMaximized()
  plt.tight_layout()
  plt.show()


#evaluate('dd1', X_train_dd5, X_test_dd5, y_train_dd5, y_test_dd5)
evaluate('dd7', X_train_dd7, X_test_dd7, y_train_dd7, y_test_dd7)



