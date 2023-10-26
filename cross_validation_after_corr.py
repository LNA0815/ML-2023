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
test_size = 0.2

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


# X_train: Train Data (54 Beobachtungen, 81 Merkmale)
# y_train: Train target (54 Beobachtungen, 1 target)
# X_test: Test Data (6 Beobachtungen, 81 Merkmale)
# y_test: Test target (6 Beobachtungen, 1 target)
#desired_total_variance = 0.8

#%% Define variables
threshold_corr = np.arange(0.4, 0.6, 0.01)  # range of threshold from 0.1 to 1
th = []   # Array for threshold to plot it in table
shapes = []
cv = 5


#%% Create Table for evaluation
def createtable(ax, name, th, value_pcr, value_lasso, shape):
  # Erstellen eines DataFrames aus den Listen
  table1 = {'threshold': th, 'slected features': shape, 'mean': value_pcr,
          'std': value_lasso}
  df1 = pd.DataFrame(table1)

  
  ax.axis('off')  # Deaktivieren der Achsen
  table_data1 = []
  table_data1.append(df1.columns.tolist())
  table_data1.extend(df1.values.tolist())


  # Erstellen der ersten Tabelle
  cell_text1 = ax.table(cellText=df1.values, colLabels=df1.columns, loc='center')
  cell_text1.auto_set_font_size(False)
  cell_text1.set_fontsize(14)
  ax.set_title(name, fontsize=18)
  cell_text1[0, 0].set_facecolor('#C2BCE0')
  cell_text1[0, 1].set_facecolor('#C2BCE0')
  cell_text1[0, 2].set_facecolor('#C2BCE0')
  cell_text1[0, 3].set_facecolor('#C2BCE0')

    #%%

def evaluate(title, X_train):
  means = []
  stds = []
  shapes = []
  th = []
  #for every variance 
  #for desired_total_variance in variance:
  for threshold in threshold_corr:

    #%% Removing Correlated Features
    # Berechnen der Korrelationsmatrix
    correlation_matrix = np.corrcoef(X_train, rowvar=False)

    # In rowvar=False, die Merkmale sind in den Spalten des Arrays

    # Erzeugen einer booleschen Maske für die obere Dreiecksmatrix der Korrelationsmatrix
    mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)

    # Finden von stark korrelierten Merkmalspaaren
    highly_correlated = np.where((correlation_matrix > threshold) & (correlation_matrix < 1.0))

    # Entfernen der stark korrelierten Merkmale (falls gewünscht)
    X_train_corr = np.delete(X_train, highly_correlated[0], axis=1)

    print(X_train_corr.shape)
    if X_train_corr.shape[1] > 0:
      th.append(round(threshold, 2))

      #%%   # #cross-validation
      # Erstellen eines Modells (z.B. lineare Regression)
      model = LinearRegression()

      # Führen Sie eine Kreuzvalidierung durch, z.B. mit 5 Folds
      scores = cross_val_score(model, X_train_corr, y_train_dd7, cv=cv, scoring='neg_mean_squared_error')

      # Da die Kreuzvalidierung 'neg_mean_squared_error' verwendet, müssen wir die Werte negieren
      mse_scores = -scores

      # Berechnen des Durchschnitts und der Standardabweichung der MSE-Werte
      mean_mse = np.mean(mse_scores)
      std_mse = np.std(mse_scores)
      shapes.append(X_train_corr.shape)

      means.append(round(mean_mse))
      stds.append(round(std_mse))

  fig, ax = plt.subplots()
  createtable(ax, 'mse', th, means, stds, shapes)
  fig.suptitle(title)
  plt.get_current_fig_manager().window.showMaximized()
  plt.tight_layout()
  plt.show()

evaluate('dd5', X_train_dd5)
evaluate('dd7', X_train_dd7)
  