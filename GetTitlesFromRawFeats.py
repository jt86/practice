from bs4 import BeautifulSoup
import requests
import pandas as pd
page = requests.get("http://techtc.cs.technion.ac.il/techtc300/techtc300.html#data")
soup = BeautifulSoup(page.content, 'html.parser')

cols = ['id1', 'id2', 'size', 'graph_dist', 'text_dist']
info_df = pd.DataFrame(columns=cols)
for i, row in enumerate(soup.find_all('tr')[1:]):
  cells = [cell.text for cell in row.children if cell!= '\n']
#   for col, val in zip(cols,cells):
#     print(col, val)
  new_row = dict(zip(cols,cells))
  info_df = info_df.append(new_row, ignore_index=True)
info_df.head(5)

all_links = [a['href'].split('org/')[1] for a in soup.find_all('a', href=True) if 'dmoz.org/' in a['href']]
info_df['name1'] = [a for i,a in enumerate(all_links) if i%2==0]
info_df['name2'] = [a for i,a in enumerate(all_links) if i%2!=0]

info_df.to_csv('techtc_info.csv')

for i, a in enumerate(soup.find_all('td')):
  if '<td align="center"><a href="' in str(a) and 'dmoz' in str(a):
    print(i, str(a))