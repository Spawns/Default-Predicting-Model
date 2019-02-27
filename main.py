from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize
from scipy.special import expit
from statistics import median, mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

file = 'Hypothetical_Default_Data.xls'
xl = pd.ExcelFile(file)
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

# Shows the histograms for every column in order to determine the outliers
# for i in range(1,19):
#     plt.subplot(5,5,i)
#     plt.title('Ratio {}'.format(i))
#     plt.hist(df1['Ratio {}'.format(i)], bins=100)
# plt.subplots_adjust(left=0.1,right=0.9, top=0.9, bottom=0.0, wspace=0.4, hspace=0.4)
# plt.show()

def minimize_smoothening_func(params):
     a, b, c, d = params
     return sum((Ys[i] - (a + b / (1 + expit(c+d*Xs[i]))))**2 for i in range(len(Xs)))


def smoothening_func(params):
     a, b, c, d, x = params
     return a + b / (1 + expit(c+d*x))

smoothening_params_by_ratio = dict()

for x in range(1, 19):
     (_, bins) =pd.qcut(df1['Ratio {}'.format(x)],20, retbins=True, duplicates='drop')
     defaults_split = df1[['Default', 'Ratio {}'.format(x)]].groupby(pd.cut(df1['Ratio {}'.format(x)], bins, include_lowest=True)).agg('sum')
     ratio_split = df1[['Default', 'Ratio {}'.format(x)]].groupby(pd.cut(df1['Ratio {}'.format(x)], bins, include_lowest=True)).agg('count')
     Xs=list()
     Ys=list()
     for i in range(len(list(ratio_split['Ratio {}'.format(x)]))):
          Ys.append(list(defaults_split['Default'])[i]/list(ratio_split['Ratio {}'.format(x)])[i])
     for i in range(len(bins)):
          Xs.append(bins[i-1]+(bins[i]-bins[i-1])/2)
     Xs=Xs[1:]
     plt.subplot(5,5,x)
     plt.title('Ratio {}'.format(x))
     plt.scatter(Xs, Ys, s=2)
     w = minimize(minimize_smoothening_func, np.zeros(4))
     smoothening_params_by_ratio[x] = [w.x[0], w.x[1], w.x[2], w.x[3]]
     new_Ys = [smoothening_func([w.x[0], w.x[1], w.x[2], w.x[3], x]) for x in Xs]
#      plt.plot(Xs, new_Ys, color='red')
#
# plt.subplots_adjust(left=0.1,right=0.9, top=0.9, bottom=0.0, wspace=0.4, hspace=0.4)
# plt.show()

for i in range(1, 19):
    ratio_with_defaults = df1[['Ratio {}'.format(i), 'Default']]
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

skf = StratifiedKFold(n_splits=10)
X = df1.iloc[:,[2,3,4,5,6,10,16,17]]
y = df1.iloc[:, 0]
AUCs = []

for k, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)
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
    plt.subplot(2,5,k + 1)
    plt.title('Model {} - AUC = {}'.format(k + 1, round(AUC - 0.5, 2)))
    plt.plot(Xs_plot, Ys_plot)
    plt.plot([0, 1], [0, 1])
    plt.plot([0, sum(y_test) / len(y_test), 1], [0, 1, 1])

print(np.mean(AUCs))
print(np.std(AUCs))
plt.subplots_adjust(left=0.05,right=0.95, top=0.9, bottom=0.0, wspace=0.4, hspace=0.4)
plt.show()

XX = np.append(arr = np.ones((9999, 1)).astype(int), values = df1.iloc[:,1:19], axis = 1)
X_opt = XX[:,[0,2,3,4,5,6,16,17]]
y = df1.iloc[:, 0]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
