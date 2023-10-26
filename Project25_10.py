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
features_std_dd7 = StandardScaler().fit_transform(features)

#getting features only until day 5

# Liste der Spaltennamen
column_names = labels_name_test_data.ravel()
column_names= column_names.astype(str)

# Finding indices, that have dd6, dd7 or Overall in their names, so we only have the data from the days before
# this can be extended to any differentiation day
indices_to_remove = [i for i, name in enumerate(column_names) if 'dd6' in name or 'dd7' in name or 'Overall' in name]

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
test_size = 0.1

X_train_dd5, X_test_dd5, y_train_dd5, y_test_dd5 = train_test_split(X_dd5, labels, test_size=test_size, random_state=42)

X_train_dd7, X_test_dd7, y_train_dd7, y_test_dd7 = train_test_split(X_dd7, labels, test_size=test_size, random_state=42)



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
from sklearn.linear_model import Lasso
from matplotlib import gridspec


# X_train: Train Data (54 Beobachtungen, 81 Merkmale)
# y_train: Train target (54 Beobachtungen, 1 target)
# X_test: Test Data (6 Beobachtungen, 81 Merkmale)
# y_test: Test target (6 Beobachtungen, 1 target)


#%% Define variables
mses_pcrpca = []  # Array für die Mean Squared Errors (MSE)
r_squares_pcrpca = []  # Array für die R-squared (R2)
mses_lassopca = []  # Array für die Mean Squared Errors (MSE)
r_squares_lassopca = []  # Array für die R-squared (R2)
variance=np.arange(0.6, 1, 0.02)  # range of variance from 60% to 100%
alpha = 0.8  # alpha for lasso regression
threshold_corr = 0.52  # threshold for removing correlated features
th = []   # Array for threshold to plot it in table


#%%PCA
def pca(desired_total_variance, x_train, x_test):
      # Perform PCA
        pca = PCA()
        principal_components = pca.fit_transform(x_train)

        # Calculate explained variance for each principal component
        explained_variance = pca.explained_variance_ratio_

        # Cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance)

        # Find the number of principal components to reach the threshold
        num_components = np.argmax(cumulative_explained_variance >= desired_total_variance) + 1

        #Bar plot of cumulative explained variance
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
        # plt.xlabel('Number of Principal Components')
        # plt.ylabel('Cumulative explained variance')
        # plt.title(f'Cumulative explained variance (Desired: {desired_total_variance * 100}%)')
        # plt.grid()
        # plt.show()
        #plt.close()
        #plt.clf()

        # Reduce the number of main components to the desired level
        pca = PCA(n_components=num_components)
        X_train_pca = pca.fit_transform(x_train)
        X_test_pca = pca.transform(x_test)
    
        return X_train_pca, X_test_pca, num_components

#%% PCA with number of components
def pca_n_components(num_components, x_train, x_test):
    # Reduce the number of main components to the desired level
    num_components = min(num_components, x_train.shape[1])
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(x_train)
    X_test_pca = pca.transform(x_test)

    # Calculate explained variance for each principal component
    explained_variance = pca.explained_variance_ratio_

    # Cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance)

    # Bar plot of cumulative explained variance
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative explained variance')
    # plt.title(f'Cumulative explained variance (Using {num_components} Components)')
    # plt.grid()
    #plt.show()

    return X_train_pca, X_test_pca

#%% PCR
def pcr(X_train_pca, X_test_pca, y_train, y_test):
      # Linear regression with the principal components
      regression_model = LinearRegression()
      regression_model.fit(X_train_pca, y_train)

      # Predictions for test data
      y_pred = regression_model.predict(X_test_pca)

      # Model performance
      mse = mean_squared_error(y_test, y_pred)
      r_squared = r2_score(y_test, y_pred)

      return mse, r_squared
#%% Lasso Regression

def lasso(X_train_pca, X_test_pca, y_train, y_test):
    # Lasso Regression
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_pca, y_train)

    # Predictions for test data
    y_pred = lasso.predict(X_test_pca)

    # Model performance
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    return mse, r_squared

#%% Create Table for evaluation
def createtable(ax, name, num, var, value_pcr, value_lasso):
  # Erstellen eines DataFrames aus den Listen
  table1 = {'variance': var,'num of components': num, 'PCR': value_pcr, 'LASSO': value_lasso}
  df1 = pd.DataFrame(table1)

  
  ax.axis('off')  # Deaktivieren der Achsen
  table_data1 = []
  table_data1.append(df1.columns.tolist())
  table_data1.extend(df1.values.tolist())


  # Erstellen der ersten Tabelle
  cell_text1 = ax.table(cellText=df1.values, colLabels=df1.columns, loc='center')
  cell_text1.auto_set_font_size(False)
  cell_text1.set_fontsize(12)
  ax.set_title(name, fontsize=12)
  cell_text1[0, 0].set_facecolor('#C2BCE0')
  cell_text1[0, 1].set_facecolor('#C2BCE0')
  cell_text1[0, 2].set_facecolor('#C2BCE0')
  cell_text1[0, 3].set_facecolor('#C2BCE0')

#%%
def evaluate(title, X_train, X_test, Y_train, Y_test):
    
  # Removing Correlated Features
  # Berechnen der Korrelationsmatrix
  correlation_matrix = np.corrcoef(X_train, rowvar=False)
  # In rowvar=False, die Merkmale sind in den Spalten des Arrays

  # Erzeugen einer booleschen Maske für die obere Dreiecksmatrix der Korrelationsmatrix
  mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
  # Finden von stark korrelierten Merkmalspaaren
  highly_correlated = np.where((correlation_matrix > threshold_corr) & (correlation_matrix < 1.0))
  # Entfernen der stark korrelierten Merkmale (falls gewünscht)
  X_train_corr = np.delete(X_train, highly_correlated[0], axis=1)
  X_test_corr = np.delete(X_test, highly_correlated[0], axis=1)

  #%%
  mses_lassopca=[]
  r_squares_lassopca =[]
  mses_pcrpca =[]
  r_squares_pcrpca=[]
  variance_round=[]
  nums_components=[]

  for var in variance:
    # PCA
    #X_train_dd5_pca, X_test_dd5_pca = pca_n_components(30, X_train_corr_dd5, X_test_corr_dd5)
    X_train_pca, X_test_pca, num_components = pca(var, X_train_corr, X_test_corr)

    # PCR
    mse_pcrpca, r_squared_pcrpca = pcr(X_train_pca, X_test_pca, Y_train, Y_test)
    mses_pcrpca.append(int(round(mse_pcrpca)))
    r_squares_pcrpca.append(round(r_squared_pcrpca, 2))

    #Lasso regression
    mse_lassopca, r_squared_lassopca = lasso(X_train_pca, X_test_pca, Y_train, Y_test)
    mses_lassopca.append(int(round(mse_lassopca)))
    r_squares_lassopca.append(round(r_squared_lassopca, 2))

    nums_components.append(num_components)
    variance_round.append(round(var, 2))

  # PCR and Lasso without PCA
  mse_pcr, r_squared_pcr = pcr(X_train_corr, X_test_corr, Y_train, Y_test)
  mse_lasso, r_squared_lasso = lasso(X_train_corr, X_test_corr, Y_train, Y_test)

  mse_pcr_round=int(round(mse_pcr))
  mse_lasso_round=int(round(mse_lasso))
  r_squared_pcr_round=round(r_squared_pcr, 2)
  r_squared_lasso_round=round(r_squared_lasso, 2)
  

  # Erstellen einer Tabelle im Plot
  fig, ax = plt.subplots(figsize=(10, 6))
  gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 5])
  ax = []
  for i in range(2):
    for j in range(2):
        ax.append(plt.subplot(gs[i, j]))
  createtable(ax[0], 'mse', [0,0], [0,0], mse_pcr_round, mse_lasso_round)
  createtable(ax[1], 'r²', [0,0], [0,0], r_squared_pcr_round, r_squared_lasso_round)
  createtable(ax[2], 'mse with PCA', nums_components, variance_round, mses_pcrpca, mses_lassopca)
  createtable(ax[3], 'r² with PCA', nums_components, variance_round, r_squares_pcrpca, r_squares_lassopca)
  fig.suptitle(title)
  plt.get_current_fig_manager().window.showMaximized()
  plt.tight_layout()
  plt.show()


evaluate('dd5', X_train_dd5, X_test_dd5, y_train_dd5, y_test_dd5)
evaluate('dd7', X_train_dd7, X_test_dd7, y_train_dd7, y_test_dd7)



