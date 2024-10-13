import pandas as pd



#   MINE-------------------------------------------------------------------------------
class DataPreprocessor:
    def fit(self, df):
        # 对age进行中位值填补
        df['Age'].fillna(df['Age'].median(), inplace=True)
        # 对embarked进行众数填补
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        # 删除无关项
        df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        # 转换为独热编码格式
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        # 转换数据类型
        df['Pclass'] = df['Pclass'].astype('category')
        # 对性别进行映射
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

        return df





#  GPT----------------------------------------------------------------------------
# class DataPreprocessor:
#     def __init__(self, fill_age_method='mean', fill_embarked_method='mode'):
#         self.fill_age_method = fill_age_method
#         self.fill_embarked_method = fill_embarked_method
#         self.label_encoders = {}
#
#     def fit(self, df):
#         # 处理缺失值
#         if self.fill_age_method == 'mean':
#             df['Age'].fillna(df['Age'].mean(), inplace=True)
#         elif self.fill_age_method == 'median':
#             df['Age'].fillna(df['Age'].median(), inplace=True)
#
#         if self.fill_embarked_method == 'mode':
#             df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#
#         # 删除Cabin列
#         df.drop(columns=['Cabin'], inplace=True)
#
#         # 编码分类变量
#         df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
#
#         # 将Pclass转换为类别型
#         df['Pclass'] = df['Pclass'].astype('category')
#
#         return df


