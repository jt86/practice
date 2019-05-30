from GetSingleFoldData import load_dataset_from_name


dataset = load_dataset_from_name('tech',0)
print(dataset)

tech_path = '/Users/joe/Desktop/Privileged_Data/techtc300_preprocessed/Exp_1092_1110'
tech_dataset = open(tech_path+'/vectors.dat')

print(tech_dataset.readlines()[:10])