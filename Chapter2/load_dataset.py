import os
# import tarfile
# import urllib.request

# DOWNLOAD_ROOT= "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# def fetch_housing_data(housing_url, housing_path):
#     os.makedirs(housing_path, exist_ok=True)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path= housing_path)
#     housing_tgz.close()

# fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)

import pandas as pd
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()
print(housing.head())
#housing


print(housing.info())
print(housing["ocean_proximity"].value_counts())
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
#save_fig("attribute_histogram_plots")
#plt.show()

import numpy as np
def split_train_test(data, test_ratio): #创建测试集,data为待处理数据，ratio是测试机的比率
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*(test_ratio))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]#划分出训练集和测试集
    return data.iloc[train_indices], data.iloc[test_indices]

#两种简单方法实现划分：1. 第一次运行程序后保存测试集； 2. 调用permutation之前设置一个随机数种子
#缺点：都会在下次获取更新的数据集时中断
#解决方法：使用一个标识符来决定是否进入测试集eg.计算每个实力标识符的哈希值，根据哈希值小于等于最大哈希值的20%则进入测试集
#实例：
train_set, test_set = split_train_test(housing, 0.2)
print("data: "+str(len(housing)))
print("train: "+str(len(train_set)))
print("test: "+str(len(test_set)))

from zlib import crc32
#
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#1.
housing_with_id = housing.reset_index()   # adds an `index` column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
#2.
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#数据集的每一列可以按照关键字检索并且按列进行合并和计算
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
print("11111111111")
print(test_set.head())
print(train_set.head())

#然而！sklearn有现成的..
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(test_set.head())
print(train_set.head())

housing["median_income"].hist()  #.hist（）制作直方图(对于series对象的操作)


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()#同样的可以生成直方图

# 于是可以采用分层抽样了！
#
# StratifiedShuffleSplit提供分层抽样的方法，参数类似于之前sklearn的train_test_split，但StratifiedShuffleSplit是根据其中某种特性的数据进行分层抽样
#
# https://blog.csdn.net/qq_30815237/article/details/87904205
# 参数说明
#
# 参数 n_splits是将训练数据分成train/test对的组数，可根据需要进行设置，默认为10，其实就是讲数据集划分了5次，得到5组(train，test)，注意每组(train，test)都是包含了所有的数据集。
#
# 参数test_size和train_size是用来设置(train，test)中train和test所占的比例。例如：
#
# 1.提供10个数据num进行训练和测试集划分
#
# 2.设置train_size=0.8 test_size=0.2
#
# 3.train_num=num*train_size=8 test_num=num*test_size=2
#
# 4.即10个数据，进行划分以后8个是训练数据，2个是测试数据

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#split是一个空的对象，我们需要使用其中的split方法，实现分成抽样
#生成测试集和训练集
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(len(strat_train_set))
print(len(strat_test_set))

#验证分层抽样,发现是一致的
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
print(housing["income_cat"].value_counts() / len(housing))
strat_test_set.head()
# 最后，删除掉为了分层而产生的income_cat属性：使用drop函数
#
# 用法：DataFrame.drop(labels=None,axis=0, index=None, columns=None, inplace=False)
#
# 参数说明：
# labels 就是要删除的行列的名字，用列表给定。
#
# axis 默认为0，指删除行，因此删除columns时要指定axis=1；
#
# index 直接指定要删除的行
#
# columns 直接指定要删除的列
#
# inplace=False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；
#
# inplace=True，则会直接在原数据上进行删除操作，删除后无法返回。
#
# 因此，删除行列有两种方式：
#
# 1）labels=None,axis=0 的组合
#
# 2）index或columns直接指定要删除的行或列
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
strat_test_set.head()#和上面对比，income_cat没了
housing.plot(kind = "scatter", x = "longitude", y = "latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend() #增加更多参数

corr_matrix = housing.corr()
  #是一个横纵坐标是不同数值的矩阵，显示相关系数
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
# 对角线上默认为直方图
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
# 选择其中部分数据可视化
scatter_matrix(housing[attributes], figsize=(12, 8))
#save_fig("scatter_matrix_plot")
housing.plot(x = "median_house_value", y = "median_income", kind = "scatter", alpha = 0.2)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set,准备一个干净的训练集
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# SimpleImputer的实例会处理DataFrame中的缺失值
# Can only use these strategies: ['mean', 'median', 'most_frequent', 'constant']
housing_num = housing.drop("ocean_proximity", axis=1)
#要删除掉非数值的属性ocean_proximity
imputer.fit(housing_num)#fit方法将训练数据和inputer适配

print(imputer.statistics_) #返回中位数
print(housing_num.median().values)

X = imputer.transform(housing_num) #所有缺失值均被替换，但此时是array数组，所以要变回DataFrame
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)

#处理文本和分类属性ocean_proximity
housing_cat = housing[["ocean_proximity"]]#一个中括号：series，两个中括号，单列DataFrame
#housing_cat
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#将属性编码
print(ordinal_encoder.categories_)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(type(housing_cat_1hot))
#Scipy稀疏矩阵
#编码为独热向量

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
# Also, `housing_extra_attribs` is a NumPy array, we've lost the column names
# (unfortunately, that's a problem with Scikit-Learn). To recover a `DataFrame`,
#  you could run this:

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns) + ["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
#将数值转换流水线和热值函数转换后的数组合并
housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression

#labelss = housing_labels.values

lin_reg = LinearRegression()
#print(labelss)
#lin_reg.fit(housing_prepared, labelss)
lin_reg.fit(housing_prepared, housing_labels)
# 创建线性回归模型，并且训练
# lin_reg.fit() 方法，后面两个参数可以是DataFrame，Series或者Array

# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

from sklearn.tree import DecisionTreeRegressor
# DecisionTreeRegressor：在决策树中找到更加复杂的非线性关系

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(lin_mse)
print(tree_rmse)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
# 参数设置： 模型；拟合参数1;拟合参数2; 策略;重复组数
# 返回的score是负值，要取负，负值中越大越好，正值中越小越好
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())  # 平均值


display_scores(tree_rmse_scores)

scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("forest_rmse"+str(forest_rmse))

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimaators':[3,10],'max_features':[2,3,4]}
]


forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
# 五折交叉
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)

plt.show()
# alpha参数，设置透明度