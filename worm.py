import pandas as pd

url = "https://www.basketball-reference.com/wnba/years/2025_advanced.html"
tables = pd.read_html(url)
adv = tables[0]

cols = ["Player","Tm","WS/40","TS%","USG%","AST%","TRB%","DBPM"]
df = adv[cols]
print(df)
