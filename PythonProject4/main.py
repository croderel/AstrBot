# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')


# 2. 数据预处理
def preprocess_data(data):
    # 处理缺失值
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # 特征工程
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0

    # 提取称呼
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # 转换分类变量
    title_encoder = LabelEncoder()
    data['Title'] = title_encoder.fit_transform(data['Title'])

    sex_encoder = LabelEncoder()
    data['Sex'] = sex_encoder.fit_transform(data['Sex'])

    embarked_encoder = OneHotEncoder(sparse=False)
    embarked_encoded = embarked_encoder.fit_transform(data['Embarked'].values.reshape(-1, 1))
    data = pd.concat([data, pd.DataFrame(embarked_encoded, columns=['Embarked_C', 'Embarked_Q', 'Embarked_S'])]

                     , axis=1)

    # 删除无用特征
    drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked']
    data.drop(drop_features, axis=1, inplace=True)

    return data


# 预处理训练集和测试集
X = preprocess_data(train_data.drop('Survived', axis=1))
y = train_data['Survived']
X_test = preprocess_data(test_data)

# 3. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 模型训练（使用随机森林）
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators':,
'max_depth':,
'min_samples_split':
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 5. 模型评估
val_predictions = best_model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
print(classification_report(y_val, val_predictions))

# 6. 生成预测结果
test_predictions = best_model.predict(X_test)

# 7. 保存结果
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)