

my_string = 'RFE & 101 & 7 & 187 & 63 & 7 & 225 & 71 & 12 & 212 \\ \hline ANOVA & 72 & 9 & 214 & 36 & 7 & 252 & 71 & 9 & 215 \\ BAHSIC & 76 & 4 & 215 & 36 & 7 & 252 & 64 & 5 & 226 \\ CHI$^2$ & 70 & 9 & 216 & 37 & 7 & 251 & 76 & 9 & 210 \\ MI & 93 & 4 & 198 & 44 & 8 & 243 & 92 & 5 & 198 \\'
new_string = ''
for item in my_string.split(' '):
    print (item)
    if item.isnumeric():
        item= '{:.1f}\%'.format((int(item)/295*100))
    new_string+=' {} '.format(item)
print(new_string)