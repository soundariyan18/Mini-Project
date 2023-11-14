# Mini-Project

## ANALYSIS OF THE DETAILS OF A PERSON

## Aim:
Analysis of the details of a person.

## ALGORITHM:
Step:1 Importing necessary packages.
Step:2 Read the data set.
Step:3 Execute the methods.
Step:4 Run the program.
Step:5 Get the output.

## CODE AND OUTPUT:
```
import pandas as pd
df = pd.read_csv("addresses.csv")
df.head(4)

```
![282276037-f0f55571-bad6-4ba8-ab6c-c7c76ed18fe4](https://github.com/soundariyan18/Mini-Project/assets/119393307/2fe3e72b-550c-43c2-8532-7597c3b63721)
```
df.info()

```
![282276052-470ecdcd-aa18-451e-9863-a7949b67a4f5](https://github.com/soundariyan18/Mini-Project/assets/119393307/3533dcb6-55b8-4f40-ae97-6b10ddebcd98)
```

df.dropna(how='all').shape
```
![282276061-38665a05-a4f9-4619-985f-013c8d280419](https://github.com/soundariyan18/Mini-Project/assets/119393307/76a6332d-e36b-491f-84ca-1ff5998f9b61)
```

df.fillna(0)
```
![282276064-929eec23-1a02-4da0-ab7d-aa1e8f021426](https://github.com/soundariyan18/Mini-Project/assets/119393307/5c97aae9-f197-4998-b2d9-a7cfe2e3ceec)
```

df.fillna(method='bfill')
```
![282276067-4ae410e4-8a37-4613-8d5d-a63e8fa028d8](https://github.com/soundariyan18/Mini-Project/assets/119393307/01dedecb-5e23-41b3-bcb6-518c3eab2697)
```

df.duplicated()
```
![282276074-c499075c-b2ca-4ac7-bb81-21e1d7bb13f7](https://github.com/soundariyan18/Mini-Project/assets/119393307/92218635-0a54-47cc-b549-95b17b8570cb)
```

exp = [13,23,28,12,5,9,31,26,10,19,22,24,29,4,25,30]
af=pd.DataFrame(exp)
af
```
![282276101-e274c952-a0db-40e6-bb26-731bb1605354](https://github.com/soundariyan18/Mini-Project/assets/119393307/c0308595-4352-439f-97c8-42be0048e7f8)
```

q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1

low=q1-1.5*iqr
low
```
![282276109-c9f2e941-a243-4459-baed-c098adf3b79e](https://github.com/soundariyan18/Mini-Project/assets/119393307/5627ad79-69e3-4bd6-b2c5-dc9d0edc3664)
```

high=q1+1.5*iqr
high
```
![282276117-1a40f1b1-d60b-45a5-9af4-d44a1d486d19](https://github.com/soundariyan18/Mini-Project/assets/119393307/3aa4f931-8057-452c-8de8-c7a61f466bfa)
```

sns.boxplot(data=af)
```
![282276126-0bd91475-fd69-4c6a-abcd-4d6094ea8ca9](https://github.com/soundariyan18/Mini-Project/assets/119393307/dcc6bb7d-a017-4f07-8542-0e5a13f29ec8)
```

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("addresses.csv")


plt.figure(figsize=(8, 4))
data['Desig'].value_counts().plot(kind='bar')
plt.title('Distribution of Desig')
plt.xlabel('Desig')
plt.ylabel('Count')
plt.show()
```
![282276145-6d9f61fd-e35a-4786-b8ca-a94db8d42773](https://github.com/soundariyan18/Mini-Project/assets/119393307/594be105-a0db-48cd-b965-3fb38d203d43)
```

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, hue="Desig")
plt.show()
```
![282276158-d9c6c278-8969-4bb0-94c3-2ed1823ad720](https://github.com/soundariyan18/Mini-Project/assets/119393307/5810becb-956e-4bd4-bc98-d7f041f13bb9)
```

correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```
![282276382-d6657d5c-7cba-4816-99ef-97371f008c35](https://github.com/soundariyan18/Mini-Project/assets/119393307/aa465944-7980-419c-9a81-3e71b394ea0b)
```

import pandas as pd
from sklearn.preprocessing import StandardScaler

numerical_features = ['ID']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
data
```
![282276193-e4d70e65-a059-4410-8053-8da823eac1f7](https://github.com/soundariyan18/Mini-Project/assets/119393307/a07b9e92-9dfc-4589-817b-66e913081e66)
```

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
columns_to_scale = ['ID']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data
```
![282276216-0a54f657-4e63-4f9a-bbc4-5fc82b5767de](https://github.com/soundariyan18/Mini-Project/assets/119393307/f2099bbc-8ab9-4b2b-bceb-2bb9d5b3afb8)
```

data.skew()
```
![282276253-c1835689-6cff-4fea-b7d0-0eb062ad896f](https://github.com/soundariyan18/Mini-Project/assets/119393307/c07d828a-c951-40c0-b5f4-9cc4f38381a7)
```

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np

np.log(df["ID"])
```
![282276268-648094f3-8ce4-4a44-9b25-85eed638ea1f](https://github.com/soundariyan18/Mini-Project/assets/119393307/8e1e26b1-fb6c-49f4-85e3-aedf3d1050a0)
```

sm.qqplot(df['ID'],line='45')
plt.show()
```
![282276291-81a92361-4689-40ab-aba7-bb63ab267a4d](https://github.com/soundariyan18/Mini-Project/assets/119393307/d02a983c-8a51-4b98-839c-fec96135e8bf)
```

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
plt.hist(data['ID'], bins=10, color='skyblue', edgecolor='black')
plt.title('Position')
plt.xlabel('ID')
plt.ylabel('Frequency')
plt.show()
```
![282276319-9257423d-d348-48c8-a391-dfd4dfe9eff6](https://github.com/soundariyan18/Mini-Project/assets/119393307/d0d7faec-7b11-4e2e-9c0d-27116f39b048)
```

plt.figure(figsize=(8, 4))
sns.boxplot(data=data, x='ID', color='lightcoral')
plt.title('Position Boxplot')
plt.xlabel('ID')
plt.show()
```
![282276334-059fb79a-629d-4760-a2fa-15c72a02bd47](https://github.com/soundariyan18/Mini-Project/assets/119393307/5195ea30-c6d0-4dee-8578-ca36882d3bd4)
```

plt.figure(figsize=(10, 4))
sns.countplot(data=data, x='Desig', palette='Set2')
plt.title('Desig Counts')
plt.xlabel('Desig')
plt.ylabel('Count')
plt.show()
```
![282276353-148525f2-f194-4dde-a262-a8e159dd3a6e](https://github.com/soundariyan18/Mini-Project/assets/119393307/718d30b2-84df-414c-9697-349cda215742)
```


## Result:
Hence the program to analyze the data set using data science is applied sucessfully.
