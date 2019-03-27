import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from collections import defaultdict
from sklearn import linear_model
import numpy as np
from scipy.optimize import minimize
from math import exp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm


xl = pd.ExcelFile('Hypothetical_Default_Data.xls')
df1 = xl.parse('All')

#Shows the Descriptive Statistics for the data
# descriptive_statistics= {
#     'Mean': [],
#     'Median': [],
#     'Skewness': [],
#     'Kurtosis': []}
# for i in range(1,19):
#     descriptive_statistics['Mean'].insert(i-1, mean(df1['Ratio {}'.format(i)]))
#     descriptive_statistics['Median'].insert(i-1, median(df1['Ratio {}'.format(i)]))
#     descriptive_statistics['Skewness'].insert(i-1, skew(df1['Ratio {}'.format(i)]))
#     descriptive_statistics['Kurtosis'].insert(i-1, kurtosis(df1['Ratio {}'.format(i)]))
# pprint.PrettyPrinter(indent=1).pprint(descriptive_statistics)

# # Outlier detection
# df1 = df1[(np.abs(stats.zscore(df1.loc[:, df1.columns != 'Default'])) < 3).all(axis=1)]

# # Shows the histograms for every column in order to determine the outliers
# for i in range(1,19):
#     plt.subplot(5,5,i)
#     plt.title('Ratio {}'.format(i))
#     plt.hist(df1['Ratio {}'.format(i)], bins=100)
# plt.subplots_adjust(left=0.1,right=0.9, top=0.9, bottom=0.0, wspace=0.4, hspace=0.4)
# plt.show()

number_of_observations_in_bucket = 500

x_per_ratio = defaultdict(list)
y_per_ratio = defaultdict(list)

for i in range(1, 19):
    ratio_with_defaults = df[['Ratio {}'.format(i), 'Default']].sort_values(by='Ratio {}'.format(i))
    ratio = ratio_with_defaults['Ratio {}'.format(i)]
    defaults = ratio_with_defaults['Default']
    for j in range(0, len(ratio), number_of_observations_in_bucket):
        bucket_ratio = ratio[j:j+number_of_observations_in_bucket]
        bucket_defaults = defaults[j:j+number_of_observations_in_bucket]
        bucket_lower_boundary = bucket_ratio.iloc[0]
        bucket_upper_boundary = bucket_ratio.iloc[-1]
        bucket_mean = mean(bucket_ratio)
        defaults_in_observation = sum(bucket_defaults)
        default_rate = defaults_in_observation / len(bucket_defaults)
        x_per_ratio[i].append(bucket_mean)
        y_per_ratio[i].append(default_rate)

def minimize_smoothehing_func(params):
    a, b, c, d = params
    return sum((Ys[i] - (a + b / (1 + exp(c+d*Xs[i]))))**2 for i in range(len(Xs)))

def smoothening_func(params):
    a, b, c, d, x = params
    return a + b / (1 + exp(c+d*x))

smoothening_params_by_ratio = dict()

for i in range(1, 19):
    plt.subplot(5,5,i)
    plt.title('Ratio {}'.format(i))
    Xs = x_per_ratio[i]
    Ys = y_per_ratio[i]
    w = minimize(minimize_smoothehing_func, np.zeros(4))
    smoothed_Ys = [smoothening_func([w.x[0], w.x[1], w.x[2], w.x[3], x]) for x in Xs]
    smoothening_params_by_ratio[i] = [w.x[0], w.x[1], w.x[2], w.x[3]]
#     plt.scatter(Xs, Ys, s=2)
#     plt.ylim(0 - 0.005, max(Ys) + 0.005)
#     plt.plot(Xs, smoothed_Ys, color='red')
    
# plt.subplots_adjust(left=0.1,right=0.9, top=0.9, bottom=0.0, wspace=0.4, hspace=0.4)
# plt.show()

for i in range(1, 19):
    ratio_with_defaults = df[['Ratio {}'.format(i), 'Default']]
    ratio = ratio_with_defaults['Ratio {}'.format(i)]
    defaults = ratio_with_defaults['Default']
    number_of_defaults = sum(defaults)
    number_of_all = len(defaults)
    ratio_smoothening_params = smoothening_params_by_ratio[i]
    PD_probabilities_defaults = []

    for j in range(len(ratio)):
        specific_ratio_PD = smoothening_func(ratio_smoothening_params + [ratio[j]])
        PD_probabilities_defaults.append((specific_ratio_PD, defaults[j]))

    Xs_plot = [0]
    Ys_plot = [0]
    current_y = 0
    current_x = 0
    AUC = 0

    for j, probability_default in enumerate(sorted(PD_probabilities_defaults, reverse=True)):
        probability, default = probability_default
        current_x += 1 / number_of_all
        Xs_plot.append(current_x)
        current_y = current_y + 1 / number_of_defaults if default else current_y
        Ys_plot.append(current_y)
        AUC += (current_y * 1 / number_of_all)

#     plt.subplot(5,5,i)
#     plt.title('Ratio {} - AUC = {}'.format(i, round(AUC - 0.5, 2)))
#     plt.plot(Xs_plot, Ys_plot)
#     plt.plot([0, 1], [0, 1])
#     plt.plot([0, number_of_defaults / len(defaults), 1], [0, 1, 1])
    
# plt.subplots_adjust(left=0.05,right=0.95, top=0.9, bottom=0.0, wspace=0.4, hspace=0.4)    
# plt.show()

skf = StratifiedKFold(n_splits=20)
X = df.iloc[:,[2,3,4,5,6,10,16,17]]
y = df.iloc[:, 0]

for column in X:
    smoothening_params = smoothening_params_by_ratio[int(column.split(' ')[1])]
    for index in range(len(X[column])):
        X[column][index] = smoothening_func(smoothening_params + [X[column][index]])

AUCs = []

for k, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(X_train, y_train)
    predicted_probabilities = clf.predict_proba(X_test)
    predicted_with_actual = []

    for i in range(len(predicted_probabilities)):
        predicted_with_actual.append((predicted_probabilities[i][1], y_test.iloc[i]))
        
    Xs_plot = [0]
    Ys_plot = [0]
    current_y = 0
    current_x = 0
    AUC = 0
    
    for j, probability_default in enumerate(sorted(predicted_with_actual, reverse=True)):
        probability, default = probability_default
        current_x += 1 / len(X_test)
        Xs_plot.append(current_x)
        current_y = current_y + 1 / sum(y_test) if default else current_y
        Ys_plot.append(current_y)
        AUC += (current_y * 1 / len(X_test))
        
    AUCs.append(AUC - 0.5)
#     plt.subplot(4,5,k + 1)
#     plt.title('Model {} - AUC = {}'.format(k + 1, round(AUC - 0.5, 2)))
#     plt.plot(Xs_plot, Ys_plot)
#     plt.plot([0, 1], [0, 1])
#     plt.plot([0, sum(y_test) / len(y_test), 1], [0, 1, 1])

# print(np.mean(AUCs))
# print(np.std(AUCs))
# plt.subplots_adjust(left=0.05,right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)    
# plt.show()

XX = np.append(arr = np.ones((9999, 1)).astype(int), values = df.iloc[:,1:19], axis = 1)
X_opt = XX[:,[0,2,3,4,5,6,16,17]]
y = df.iloc[:, 0]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()