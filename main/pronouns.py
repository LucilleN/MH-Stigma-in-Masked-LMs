# Comparing He vs She
from transformers import pipeline
import pandas as pd
from utils import *
import argparse
import numpy as np
from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats

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

# Effect Size


def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

# Top-K Function


def get_top_k(template, male_mask, female_mask):
    male = []
    female = []

    for diagnosis in diagnoses:
        prompt = template.replace("[diagnosis]", diagnosis)
        output_list = nlp_fill(prompt)

        female_exist = 0
        male_exist = 0
        # print(diagnosis)
        # print(output_list)
        for i in range(TOP_K):

            if output_list[i]['token_str'] == female_mask:
                female.append(output_list[i]['score'])
                female_exist = 1
            if output_list[i]['token_str'] == male_mask:
                male.append(output_list[i]['score'])
                male_exist = 1

        if female_exist == 0:
            female.append(0)
        if male_exist == 0:
            male.append(0)

    assert len(diagnoses) == len(female)
    assert len(diagnoses) == len(male)
    # print (female)
    # print (male)

    return male, female

# Target Token Probability


def get_target_probability(template, target_token):
    outputs = []

    for diagnosis in diagnoses:
        prompt = template.replace("[diagnosis]", diagnosis)
        output_list = nlp_fill(prompt, targets=target_token)
        # score = output_list[0]['score']
        # scores_by_diagnosis.append(score)
        outputs.append(output_list[0])

    return outputs

# Print Stats


def print_stats(male, female):
    male_mean = mean(male)
    female_mean = mean(female)
    print(f"Mean: male={male_mean}, female={female_mean}")
    print(f"T-test: {stats.ttest_rel(female, male)}")
    print(f"Effect Size: {cohend(female, male)}")
    return male_mean, female_mean

# Plot male and female


def plot_male_and_female(template, male_mask, female_mask, male, female):
    male_df = pd.DataFrame({'Diagnosis': diagnoses, 'Subject': [
                           male_mask]*len(male), 'Probability': male})
    female_df = pd.DataFrame({'Diagnosis': diagnoses, 'Subject': [
                             female_mask]*len(female), 'Probability': female})
    df = pd.concat([female_df, male_df])

    ax = sns.scatterplot(data=df, x='Diagnosis',
                         y='Probability', hue='Subject')
    locs, labels = plt.xticks()

    plt.setp(labels, rotation=35, ha='right')
    plt.title(f"Probability of Predicted Subject \n {template}")
    # plt.figure(dpi=90)

    filename = template.replace(
        "<mask>", f"{female_mask.strip()}-{male_mask.strip()}")
    # print(f"{filename}.pdf")
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")


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
    all_df = add_to_df(male, female, template)  # , all_df)
    # print (all_df)

# Experiment 2: She/He struggles with X


def struggles_with():
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
    all_df = add_to_df(male, female, template)  # , all_df)


all_experiments = [
    has,
    is_diagnosed_with,
    struggles_with,
    # suffers_from,
    # is_in_recovery_from,
    # is_in_treatment_for,
    # is_being_treated_for,
    # sees_a_psychiatrist_for,
    # sees_a_therapist_for,
    # is_going_to_therapy_for,
    # takes_medication_for
]
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

    male, female = get_top_k(template, male_mask, female_mask)
    print(male)
    print(female)

    print("")
    male_mean, female_mean = print_stats(male=male, female=female)
    update_aggregates(male_mean, female_mean, template, treated=False)

    plot_male_and_female(template, male_mask, female_mask, male, female)
    all_df = add_to_df(male, female, template)  # , all_df)


for exp_number in exps_to_run:
    # all_experiments[exp_number]()
    template = templates[exp_number]
    run_experiment(template)
