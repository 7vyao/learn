import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from model.DataPreprocessor import DataPreprocessor
from model.logistic import MyLogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 加载数据集
train_data = pd.read_csv('data/titanic/train.csv')
test_data = pd.read_csv('data/titanic/test.csv')


# 数据预处理
preprocessor = DataPreprocessor()
train_data = preprocessor.fit(train_data)
test_data = preprocessor.fit(test_data)

# 分割特征和结果
x = train_data.drop('Survived', axis=1)
y = train_data['Survived']
x_test = test_data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


# 标准化特征
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 训练模型
start = time.time()
model = MyLogisticRegression()
model.fit(x_train, y_train)
print(f'Training time: {time.time() - start:.2f} seconds')

# 预测结果
y_pred_train = model.predict(x_train)
y_pred_val = model.predict(x_val)

# 评估模型
train_accuracy = accuracy_score(y_train, y_pred_train)
val_accuracy = accuracy_score(y_val, y_pred_val)

train_precision = precision_score(y_train, y_pred_train)
val_precision = precision_score(y_val, y_pred_val)

train_recall = recall_score(y_train, y_pred_train)
val_recall = recall_score(y_val, y_pred_val)

train_confusion_matrix = confusion_matrix(y_train, y_pred_train)
val_confusion_matrix = confusion_matrix(y_val, y_pred_val)

print(f'Training accuracy: {train_accuracy:.2f}')
print(f'Validation accuracy: {val_accuracy:.2f}')

print(f'Training precision: {train_precision:.2f}')
print(f'Validation precision: {val_precision:.2f}')

print(f'Training recall: {train_recall:.2f}')
print(f'Validation recall: {val_recall:.2f}')

print('Training confusion matrix:')
print(train_confusion_matrix)

print('Validation confusion matrix:')
print(val_confusion_matrix)

results = pd.DataFrame({
    'Validation Accuracy': [val_accuracy],
    'Validation Precision': [val_precision],
    'Validation Recall': [val_recall]
})
val_confusion_matrix = confusion_matrix(y_val, y_pred_val)
val_confusion_matrix_df = pd.DataFrame(val_confusion_matrix, index=['Predicted Negative', 'Predicted Positive'], columns=['Actual Negative', 'Actual Positive'])


with pd.ExcelWriter('model_evaluation_results.xlsx') as writer:
    results.to_excel(writer, sheet_name='Metrics')
    val_confusion_matrix_df.to_excel(writer, sheet_name='Validation Confusion Matrix')


