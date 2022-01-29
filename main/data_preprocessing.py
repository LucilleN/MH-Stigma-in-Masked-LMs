import pickle
import random

with open('gender_names_cisgender.pickle', 'rb') as file:
    data = pickle.load(file)

# data is a tuple of 8 sets: 
#     name_nb, 
#     name_men, 
#     name_women, 
#     name_transgender_men, 
#     name_transgender_women, 
#     name_cisgender_men, 
#     category_nb, 
#     category_LGBT
print(f'data: {type(data)} of length {len(data)}')

name_men = data[1]
print(f'  > name_men is tuple item 1, a {type(name_men)} of length {len(name_men)}') 
# print(list(name_men)[:20])

name_women = data[2]
print(f'  > name_women is tuple item 2, a {type(name_women)} of length {len(name_women)}') 
# print(list(name_women)[:20])

first_names_women = set([name.split("_")[0] for name in name_women])
print(f'first_names_women is a {type(first_names_women)} of length {len(first_names_women)}') 
print(list(first_names_women)[:20])

# i = 0
# for inner_list in data:
#     print(f'  >  tuple item {i}, a {type(inner_list)} of length {len(inner_list)}') 
#     print(random.sample(inner_list, 9))
#     i += 1