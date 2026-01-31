# 保存为 pull_wnba_skill_vector_2023.py
# 作用：将 2023 赛季 WNBA 常规赛主要球员的高级统计指标保存成 CSV。
# 输出文件：wnba_2023_skill_vector.csv

import pandas as pd
import requests
from bs4 import BeautifulSoup

YEAR = 2023
url = f"https://www.basketball-reference.com/wnba/years/{YEAR}_advanced.html"

# 模拟浏览器请求，解决 403 问题
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/119.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
response.raise_for_status()

# 使用 pandas 解析高级数据表
tables = pd.read_html(response.text)
# 选取最大行数的表格（通常是球员高级统计表）
df = max(tables, key=lambda t: t.shape[0])

# 删除重复的表头行
df = df[df['Player'] != 'Player'].copy()

# 过滤：选取每队主要轮换球员（例如出场时间 >=300 分钟或可自行调整）
if 'MP' in df.columns:
    df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
    df = df[df['MP'] >= 300]

# 统一列名
if 'Tm' in df.columns:
    df.rename(columns={'Tm': 'Team'}, inplace=True)

# 转换需要的数值列
need_cols = ['WS/40','TS%','USG%','AST%','TRB%']
for col in need_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 处理 DBPM：若有 DBPM 列，直接用；否则利用 BPM-OBPM 得到防守贡献近似
if 'DBPM' in df.columns:
    df['DBPM'] = pd.to_numeric(df['DBPM'], errors='coerce')
elif 'BPM' in df.columns and 'OBPM' in df.columns:
    df['BPM'] = pd.to_numeric(df['BPM'], errors='coerce')
    df['OBPM'] = pd.to_numeric(df['OBPM'], errors='coerce')
    df['DBPM'] = df['BPM'] - df['OBPM']
else:
    # 无 DBPM 时可使用 DWS/MP 等防守指标作为近似
    df['DBPM'] = pd.NA

# 导出所需字段
out_cols = ['Player','Team','WS/40','TS%','USG%','AST%','TRB%','DBPM']
out = df[[c for c in out_cols if c in df.columns]].copy()

out.to_csv('wnba_2023_skill_vector.csv', index=False, encoding='utf-8-sig')
print("输出完成：", len(out), "行")