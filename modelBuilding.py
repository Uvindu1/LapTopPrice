import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('laptop_price.csv', encoding='latin-1')
print(data.head(2))

##################################3################################################33
# string int kawalame thiyana ewa intiger kirima
data['Weight'] = data['Weight'].str.replace('kg','').astype('float32')
data['Ram'] = data['Ram'].str.replace('GB','').astype('int32')

#######################################################################################
# correlation balanawa
# mein laba denne int, float pamani
cor = data.corr()['Price_euros']
print(cor)
##########################################################################################3
# ek ek colom eka gena wena wenama bala ewa one hot enkoding walata gelapena leas sedima

# 1) Company
Company_value = data['Company'].value_counts()
print(Company_value)
print(len(Company_value))
#mehi colom 19 k sedimata siduweno oneHotEncoding walada, eya adu kara genimata value eka 10 yata adu Company other kiyala wenama catakary ekakata danna

def add_company(company_name):
    if (company_name == 'Samsung' or company_name =='Razer' or company_name == 'Mediacom' or company_name == 'Microsoft' or company_name == 'Xiaomi' or company_name == 'Vero' or company_name == 'Chuwi' or company_name == 'Google' or company_name == 'Fujitsu' or company_name == 'LG' or company_name == 'Huawei' ):
        return 'other'
    else:
        return company_name


data['Company'] = data['Company'].apply(add_company)
print(data['Company'].value_counts())

# 2) Product
Product_value = data['Product'].value_counts()
print(len(Product_value))
# mehi catakary 618 thiyanoo enam price ekata mehi bala pema adu lesa gena ewath karanna puluwan

# 3) TypeName
TypeName_value = data['TypeName'].value_counts()
print(TypeName_value)
# mehi ketagary 6 i, eka oneHotEncode karanna puluwan


# 4)ScreenResolution
ScreenResolution_value = data['ScreenResolution'].value_counts()
print('.....................................................')
print(ScreenResolution_value)
# mehi catagary godak atha, emanisa ain viya yuthui
# namuth mehi lap performance atha (price ekata ewa balapai), ex: IPS, FULL HD...
# Ips and Touchscreen thiyeda nedda balanna wenama colom dekaka hadamu
data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
data['IPS'] = data['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
print(data)

# 4) CPU
print("**************************************************************************************")
cpu_value = data['Cpu'].value_counts()
print(cpu_value)
# mehidi apata wedagath wenne processar model eka pamani,
# processara model ekata colom ekaka sedima
# his then walin kadala mull wachana thuna pamanak genima
data['cpu_name'] = data['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
# thamath mehi catagarys ganana wediya, one hot encoding karanna amarui
# catakary ganana adu kirima
def set_process(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    elif name.split()[0] == 'AMD':
        return 'AMD'
    else:
        return 'other'

data['cpu_name'] = data['cpu_name'].apply(set_process)

print(data['cpu_name'].value_counts())


# 5) GPU
print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
Gpu_value = data['Gpu'].value_counts()
print(Gpu_value)
# mula nama anuwa ketagary walata bedanoo
data['Gpu_name'] = data['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
print(data['Gpu_name'].value_counts())
# ARM kiyana data eken data set ekata loku balapemak nee, eka ewath karanna ona
print(data.shape)
data = data[data['Gpu_name'] != 'ARM']
print(data.shape)

# 6)
print('""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""')
OpSys_value = data['OpSys'].value_counts()
print(OpSys_value)

def set_os(input):
    if input == 'Windows 10' or input == 'Windows 7 ' or input == 'Windows 10 S':
        return 'Windows'
    elif input == 'macOS' or input == 'Mac OS X':
        return 'Mac'
    elif input == 'Linux':
        return input
    else:
        return 'other'

data['OpSys'] = data['OpSys'].apply(set_os)

print(data['OpSys'])

####################################################################################################
# wedak nethi colom ewath kirima
data = data.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
print(data.value_counts())


######################################################################################################################################
#########################################################################################################
 # data set ekama one Hot Encoding kirima

data = pd.get_dummies(data)
print(data.head(2))
 # mehi siyalu data numarykal value vi atha

###########################################################################################################
#........................................MODEL CREATE ...................................................
###########################################################################################################

x = data.drop('Price_euros', axis=1)
y = data['Price_euros']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# api model kihipayakata dada test karana nisa ekma haema ekema accuracy eka ganna function ekaka liyamu
def model_set(model):
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    print(str(model)+'----->'+str(acc))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_set(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_set(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_set(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_set(rf)

# hoda hyperperameter ekak thora genima
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[10, 50, 100], 'criterion': ['squared_error','absolute_error','poisson']}
grid_obj = GridSearchCV(estimator = rf, param_grid=parameters)
grid_fit = grid_obj.fit(x_train, y_train)
best_model = grid_fit.best_estimator_
print(best_model)


# model eka save kirima
import pickle
with open('predictor.pickle','wb') as file:
    pickle.dump(best_model,file)

