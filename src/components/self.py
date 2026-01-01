import os
import pandas as pd
'''from pathlib import Path

root = Path('.').resolve()
for p in root.rglob('*.py'):
    try:
        txt = p.read_text(encoding='utf-8')
    except Exception:
        continue
    if 'stud.csv' in txt:
        print(p)
'''

rain_data_path:str = os.path.join('artifacts','rain.csv')
print(rain_data_path)
df = pd.read_csv(r'C:\datasciencejourney\FIRST_CAPSTONE_PROJECT\src\components\artifacts\data.csv')
os.makedirs(rain_data_path,exist_ok=False)
df.to_csv(rain_data_path,index=False,header=True)
