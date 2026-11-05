import pandas as pd

df = pd.read_csv('data/raw/vpass_1월 데이터.csv', encoding='utf-8-sig')
# 글자가 깨지거나 오류 나면 encoding='cp949'

print(df.shape)          
print(df.dtypes)         
df.head()                