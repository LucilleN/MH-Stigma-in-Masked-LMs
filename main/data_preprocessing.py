import pickle
import random
import math

with open('../data/gender_names_cisgender.pickle', 'rb') as file:
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
name_women = data[2]
print(f'  > name_men is tuple item 1, a {type(name_men)} of length {len(name_men)}') 
print(f'  > name_women is tuple item 2, a {type(name_women)} of length {len(name_women)}') 
# print(list(name_men)[:20]) # Converting set to list just so we can print a few examples
# print(list(name_women)[:20])

first_names_men = [name.split("_")[0] for name in name_men]
first_names_women = [name.split("_")[0] for name in name_women]
size_first_names_men = len(first_names_men)
size_first_names_women = len(first_names_women)
print(f'first_names_men is a {type(first_names_men)} of length {size_first_names_men}') 
print(f'first_names_women is a {type(first_names_women)} of length {size_first_names_women}') 
# print(first_names_men[:20])
# print(first_names_women[:20])


def get_log_odds_score(name):
    # Log odds with Dirichlet prior (see https://journals.uic.edu/ojs/index.php/fm/article/view/4944/3863)
    # General formula: 
    #     log(count of w in first corpus / (size of first corpus - count w in first corpus)) 
    #         - log(count of w in second corpus / (size of second corpus - count w in second corpus))
    # Since we're using male corpus as first log and female corpus as second log, 
    # the more positive = more strongly male and more negative = more strongly female
    count_m = first_names_men.count(name)
    # print(f"count_m: {count_m}")
    count_w = first_names_women.count(name)
    # print(f"count_w: {count_w}")
    # adding +1 (Laplace Smoothing?) to avoid log(0) = undefined
    first_log = math.log( (count_m + 1) / (size_first_names_men - count_m) )
    second_log = math.log( (count_w + 1) / (size_first_names_women - count_w) )
    delta = first_log - second_log
    return delta

if __name__ == "__main__":
    all_names = set(first_names_men).union(set(first_names_women))
    all_names_log_odds_scores = [(get_log_odds_score(name), name) for name in all_names]
    all_names_log_odds_scores.sort(reverse=True)

    # Print the top 5 male and female names
    # print(all_names_log_odds_scores[:5])
    # print(all_names_log_odds_scores[-5:])

    # with open("../data/men_top_1000.txt", "w+") as file:
    #     for i in range(1000):
    #         score, name = all_names_log_odds_scores[i]
    #         file.write(name + "\n")
            
    # with open("../data/women_top_1000.txt", "w+") as file:
    #     for i in range(len(all_names_log_odds_scores) - 1, len(all_names_log_odds_scores) - 1001, -1):
    #         score, name = all_names_log_odds_scores[i]
    #         file.write(name + "\n")

    with open("../data/men_top_1000.txt", "w+") as file:
        for i in range(1000):
            score, name = all_names_log_odds_scores[i]
            file.write(f"{score}, {name}\n")
            
    with open("../data/women_top_1000.txt", "w+") as file:
        for i in range(len(all_names_log_odds_scores) - 1, len(all_names_log_odds_scores) - 1001, -1):
            score, name = all_names_log_odds_scores[i]
            file.write(f"{score}, {name}\n")
        