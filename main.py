# Здесь должен быть твой код
#комментировать весь код
import pandas as pd
from sklearn.model_selection import train_test_split#функция для разбиения исходного набора данных на выборки для обучения и тестирования модели
from sklearn.preprocessing import StandardScaler#класс для стандартизации показателей
from sklearn.neighbors import KNeighborsClassifier#класс для создания и обученимя модели
from sklearn.metrics import confusion_matrix, accuracy_score#функции для оценки точности работы моджели
df = pd.read_csv('titanic.csv')

def fill_age(row):#это serious конкретного пассажира 
    if pd.isnull(row['Age']):
        if row['Pclass'] ==1:
            return age_1
        if row['Pclass'] ==2:
            return age_2
        return age_3
    return row['Age']


def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0

print(df.head())

print(df.groupby('Sex')['Survived'].mean())#среднее количество выживших
print(df.pivot_table(index = 'Survived',
                    columns = 'Pclass',
                    values= 'Age',
                    aggfunc = 'median'))
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1,inplace= True)#удаляем лищние столбики(id, имя, номер билета, номер каюты, )
print(df.info())

print(df['Embarked'].value_counts())#сколько пассажиров зашли на корабль с каждого порта

df['Embarked'].fillna('S' , inplace = True)#заполняем пустые значения буквой S тех пассажиров кто зашел с первого порта

print(df.groupby('Pclass')['Age'].median())

age_1 = df[df['Pclass'] == 1]['Age'].median()#средний возраст виживших 
age_2 = df[df['Pclass'] == 2]['Age'].median()#средний возраст выживших
age_3 = df[df['Pclass'] == 3]['Age'].median()#средний возраст выживших


df['Age'] = df.apply(fill_age, axis =1)#заполняем пустые значения в столбце возраст на средний возраст всех пассажиров 
print(df.info())


df['Sex'] = df['Sex'].apply(fill_sex)#заполняем столбик с полом 1=м и 0=ж

print(pd.get_dummies(df['Embarked']))#преобразцет одну категореальную переменных в несколько фиктивных

df[list(pd.get_dummies(df['Embarked']).columns)]= pd.get_dummies(df['Embarked'])# список из столбцов для каждого знаения фиктивной переменной

df.drop('Embarked', axis =1, inplace = True)#удаляем столбик откуда сел пассажир

X = df.drop('Survived', axis =1)#Данные о пассажирах
y = df['Survived']#целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)#функция разбивает данные случайным образом на обучающие и тестовые

sc = StandardScaler()#стандартизация значений
X_train = sc.fit_transform(X_train)#метод для обучающих данных
X_test = sc.transform(X_test)#метод для тестовых данных

classifier = KNeighborsClassifier(n_neighbors = 3)#количество соседей для пассажиров
classifier.fit(X_train, y_train)#подбирает признаки из набора обучающих данных

y_pred = classifier.predict(X_test)#предсказывает значение для тестовых данных

print(accuracy_score(y_test, y_pred) * 100)#точность предсказания в процентах






