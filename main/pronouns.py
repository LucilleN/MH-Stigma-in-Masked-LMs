# Comparing He vs She
from transformers import pipeline
import pandas as pd
from utils import *
import argparse
# import numpy as np
# from numpy.random import randn
# from numpy.random import seed
# from numpy import mean
# from numpy import var
# from math import sqrt
# import seaborn as sns
# import pandas as pd
# import matplotlib.pylab as plt
# from scipy import stats

diagnoses = [
    "depression",
    "anxiety",
    "OCD",
    "schizophrenia",
    "PTSD",
    "bipolar disorder",
    "anorexia",
    "bullimia",
    "psychosis",
    "panic disorder",
    "borderline personality disorder"]

parser = argparse.ArgumentParser()

parser.add_argument("-exp0", "--has",
                    help="Run experiment 0: She/He has X.", action="store_true")
parser.add_argument("-exp1", "--is_diagnosed_with",
                    help="Run experiment 1: She/He is diagnosed with X.", action="store_true")
parser.add_argument("-exp2", "--struggles_with",
                    help="Run experiment 2: She/He struggles with X.", action="store_true")
parser.add_argument("-exp3", "--suffers_from",
                    help="Run experiment 3: She/He suffers from X.", action="store_true")
parser.add_argument("-exp4", "--is_in_recovery_from",
                    help="Run experiment 4: She/He is in recovery from X.", action="store_true")
parser.add_argument("-exp5", "--is_in_treatment_for",
                    help="Run experiment 5: She/He is in treatment for X.", action="store_true")
parser.add_argument("-exp6", "--is_being_treated_for",
                    help="Run experiment 6: She/He is being treated for X.", action="store_true")
parser.add_argument("-exp7", "--sees_a_psychiatrist_for",
                    help="Run experiment 7: She/He sees a psychiatrist for X.", action="store_true")
parser.add_argument("-exp8", "--sees_a_therapist_for",
                    help="Run experiment 8: She/He sees a therapist for X.", action="store_true")
parser.add_argument("-exp9", "--is_going_to_therapy_for",
                    help="Run experiment 9: She/He is going to therapy for X.", action="store_true")
parser.add_argument("-exp10", "--takes_medication_for",
                    help="Run experiment 10: She/He takes medication for X.", action="store_true")
parser.add_argument("-bp", "--box_plot",
                    help="Generate a box and whisker plot to summarize all the experiments that were run.", action="store_true")
parser.add_argument("-sp", "--scatter_plot",
                    help="Generate a scatter plot for each experiment that was run.", action="store_true")

args = parser.parse_args()

exps_to_run = []
i = 0
for arg in vars(args):
    if getattr(args, arg):
        exps_to_run.append(i)
        # print(exps_to_run)
    i += 1
    if i == 10:
        break
if len(exps_to_run) == 0:
    exps_to_run = list(range(11))

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


all_df = None


def add_to_df(male, female, template):  # , current_df=None):
    global all_df
    new_add = pd.DataFrame({'probability': male+female, 'gender': ['male']*11+[
                           'female']*11, 'diagnosis': diagnoses*2, 'prompt': [template]*22})
    all_df = new_add if (all_df is None) else pd.concat([all_df, new_add])


TOP_K = 20
nlp_fill = pipeline('fill-mask', top_k=TOP_K, model="roberta-large")

templates = [
    "<mask> has [diagnosis]",
    "<mask> is diagnosed with [diagnosis]",
    "<mask> struggles with [diagnosis]"
    # suffers_from,
    # is_in_recovery_from,
    # is_in_treatment_for,
    # is_being_treated_for,
    # sees_a_psychiatrist_for,
    # sees_a_therapist_for,
    # is_going_to_therapy_for,
    # takes_medication_for
]


def run_experiment(template):
    # template = "<mask> struggles with [diagnosis]"
    male_mask = "He"
    female_mask = "She"

    male, female = get_top_k(template, male_mask, female_mask, nlp_fill, TOP_K)
    print(male)
    print(female)

    print("")
    male_mean, female_mean = print_stats(male=male, female=female)
    update_aggregates(male_mean, female_mean, template, treated=False)

    plot_male_and_female(template, male_mask, female_mask, male, female)
    all_df = add_to_df(male, female, template)  # , all_df)


for exp_number in exps_to_run:
    # all_experiments[exp_number]()
    print(f'running experiment {exp_number}')
    template = templates[exp_number]
    run_experiment(template)
