# Comparing He vs She
from transformers import pipeline
import pandas as pd
from utils import *

# Aggregates for the following She/He experiments

female_untreated, female_treated, female_total = {}, {}, {}
male_untreated,   male_treated,   male_total = {}, {}, {}


def update_aggregates(male_mean, female_mean, template, treated=False):
    female_total[template] = female_mean
    male_total[template] = male_mean

    if treated:
        female_treated[template] = female_mean
        male_treated[template] = male_mean
    else:
        female_untreated[template] = female_mean
        male_untreated[template] = male_mean

    # print("Updated Aggregates: ")
    # print(female_total)
    # print(female_treated)
    # print(female_untreated)


def add_to_df(male, female, template, current_df=None):
    new_add = pd.DataFrame({'probability': male+female, 'gender': ['male']*11+[
                           'female']*11, 'diagnosis': diagnoses*2, 'prompt': [template]*22})
    if current_df is None:
        return new_add
    else:
        return pd.concat([current_df, new_add])


TOP_K = 20
nlp_fill = pipeline('fill-mask', top_k=TOP_K, model="roberta-large")

# Experiment 0: She/He has X


def has():
    template = "<mask> has [diagnosis]"
    male_mask = "He"
    female_mask = "She"

    male, female = get_top_k(template, male_mask, female_mask)
    print(male)
    print(female)

    print("")
    male_mean, female_mean = print_stats(male=male, female=female)
    update_aggregates(male_mean, female_mean, template, treated=False)

    plot_male_and_female(template, male_mask, female_mask, male, female)
    all_df = add_to_df(male, female, template)
    print(all_df)

# Experiment 1: She/He is diagnosed with X


def is_diagnosed_with():
    template = "<mask> is diagnosed with [diagnosis]"
    male_mask = "He"
    female_mask = "She"

    male, female = get_top_k(template, male_mask, female_mask)
    print(male)
    print(female)

    print("")
    male_mean, female_mean = print_stats(male=male, female=female)
    update_aggregates(male_mean, female_mean, template, treated=False)

    plot_male_and_female(template, male_mask, female_mask, male, female)
    all_df = add_to_df(male, female, template, all_df)
    # print (all_df)

# Experiment 2: She/He struggles with X


def struggles_wih():
    template = "<mask> struggles with [diagnosis]"
    male_mask = "He"
    female_mask = "She"

    male, female = get_top_k(template, male_mask, female_mask)
    print(male)
    print(female)

    print("")
    male_mean, female_mean = print_stats(male=male, female=female)
    update_aggregates(male_mean, female_mean, template, treated=False)

    plot_male_and_female(template, male_mask, female_mask, male, female)
    all_df = add_to_df(male, female, template, all_df)
