import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import copy
import random
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, utils
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler



#importing data
pd.set_option('display.max_columns', 13)
pd.set_option('display.width', None)
data = pd.read_csv('cars.csv')

print(data.head())
print(data.info())
print(data.describe())
print(data.describe(include=['object']))

#data cleansing
data['TRANSMISSION'] = data['TRANSMISSION'].fillna(data['TRANSMISSION'].mode()[0])
data['ENGINESIZE'] = data['ENGINESIZE'].fillna(data['ENGINESIZE'].mean())
data['FUELTYPE'] = data['FUELTYPE'].fillna(data['FUELTYPE'].mode()[0])

# input-output dependencies

sb.heatmap(data.corr(), annot=True, square=True, fmt='.2f')
plt.show()

plt.plot(data.MODELYEAR, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of year')
plt.show()
plt.plot(data.MAKE, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 on the brand')
plt.show()
plt.plot(data.MODEL, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of model')
plt.show()
plt.plot(data.VEHICLECLASS, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of vehicleclass')
plt.show()
plt.plot(data.ENGINESIZE, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of enginesize')
plt.show()
plt.plot(data.FUELCONSUMPTION_CITY, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of fuelconsuption_city')
plt.show()
plt.plot(data.FUELCONSUMPTION_HWY, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of FUELCONSUMPTION_HWY')
plt.show()
plt.plot(data.FUELCONSUMPTION_COMB, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of FUELCONSUMPTION_COMB')
plt.show()
plt.plot(data.FUELCONSUMPTION_COMB_MPG, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of FUELCONSUMPTION_COMB_MPG')
plt.show()
plt.plot(data.FUELTYPE, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of fueltype')
plt.show()
plt.plot(data.TRANSMISSION, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of transmission')
plt.show()
plt.plot(data.CYLINDERS, data.CO2EMISSIONS, '*')
plt.title('dependency CO2 of cylinders')
plt.show()


#boxplot
# plt.figure(figsize=(8, 5))
# sb.boxplot(x='MAKE', y='CO2EMISSIONS', data=data, palette='rainbow')
# plt.title("CO2 emission dependence od brand")
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sb.boxplot(x='MODEL', y='CO2EMISSIONS', data=data, palette='rainbow')
# plt.title("CO2 emission dependence od model")
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sb.boxplot(x='TRANSMISSION', y='CO2EMISSIONS', data=data, palette='rainbow')
# plt.title("CO2 emission dependence od transmission")
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sb.boxplot(x='VEHICLECLASS', y='CO2EMISSIONS', data=data, palette='rainbow')
# plt.title("CO2 emission dependence od vehicleclass")
# plt.show()
#
# plt.figure(figsize=(8, 5))
# sb.boxplot(x='FUELTYPE', y='CO2EMISSIONS', data=data, palette='rainbow')
# plt.title("CO2 emission dependence od fueltype")
# plt.show()


#encoding categorical data
ohe = preprocessing.OneHotEncoder(dtype=int, sparse_output=False)
fueltype = ohe.fit_transform(data['FUELTYPE'].to_numpy().reshape(-1, 1))
data.drop(columns=['FUELTYPE'],  inplace=True)
data = data.join(pd.DataFrame(data=fueltype, columns=ohe.get_feature_names_out(['fueltype'])))


#feature extraction
#unrelevant features: modelyear, model, make, transmission, vehichleclass
#relevant features: everything else


data_train = data.loc[:, ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'fueltype_D','fueltype_E', 'fueltype_X', 'fueltype_Z', 'CYLINDERS' ]]
true_output = data['CO2EMISSIONS']

#scaling data

scaler = MinMaxScaler()
data.train = scaler.fit(data_train)



#making model

X_train, X_test, Y_train, Y_test = train_test_split(data_train, true_output, train_size=0.8, random_state=14, shuffle=True)
X_train.reset_index(drop=True, inplace=True)
Y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)


weight_vector = np.random.randn(X_train.shape[1])
bias = 0
learning_rate = 0.0035

def loss_function(x,y,y_predicted):
    s = 0
    for i in range(len(y)):
        s += (y_predicted[i]-y[i])**2
    loss = s/(2*len(y))
    return loss

def prediction(w,x,bias):
    y_list = []
    for i in range(len(x)):
        y_list.append(w@x.iloc[i]+bias)
    return np.array(y_list)

def derivate_wj(x,y,y_pred,j):
    sum=0
    for i in range(len(x)):
        p = x.iloc[i, j]
        sum += -x.iloc[i, j]*(y[i]-y_predicted[i])
    return sum/(len(x))


def derivate_b(x, y, y_pred):
    sum = 0
    for i in range(len(y)):
        sum += -(y[i] - y_pred[i])
    return (1 / len(y)) * sum

epoch=25

loss=[]
for m in range(epoch):
     print(m)
     y_predicted = prediction(weight_vector, X_train, bias)

     for j in range(len(weight_vector)):
        weight_vector[j] = weight_vector[j] - learning_rate * derivate_wj(X_train, Y_train, y_predicted, j)
     bias = bias - learning_rate * derivate_b(X_train, Y_train, y_predicted)
     broj = loss_function(X_train, Y_train, y_predicted)
     loss.append(broj)


print('Weight wector w: ', weight_vector)
print('bias: ', bias)

plt.show()
print(loss_function(X_train, Y_train, y_predicted))
y_predicted = prediction(weight_vector, X_test, bias)
for i in range(y_predicted.shape[0]):
    print(y_predicted[i] , " ", Y_test[i])

def score(Y_test,y_predicted):

    u = ((Y_test - y_predicted) ** 2).sum()

    v = ((Y_test - Y_test.mean()) ** 2).sum()

    score = 1 - (u / v)
    return score

print('Model score: ', score(Y_test,y_predicted))
print('============================================================================')
#ugradjen model

#Ugradjen model
X_train, X_test, Y_train, Y_test = train_test_split(data_train, true_output, train_size=0.7, random_state=14, shuffle=True)
reg = linear_model.LinearRegression()
model = reg.fit(X_train, Y_train)
print('coef: ', reg.coef_)
prediction = model.predict(X_test)
ser_pred = pd.Series(data=prediction, name='predicted', index=X_test.index)
res_df = pd.concat([X_test, Y_test, ser_pred], axis=1)
print(res_df.head())
print('Built-in model score: ', model.score(X_test, Y_test))
model.score()

