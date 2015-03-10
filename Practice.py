number_of_features=42
tuple  = [1, 42, 5]
range_of_top_feats = range(*tuple)

for n_top_feats in range(*tuple)+[number_of_features]:
     print n_top_feats