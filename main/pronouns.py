# Comparing He vs She
from transformers import pipeline
import pandas as pd
from utils import *
import argparse


parser = argparse.ArgumentParser(
    usage="To run all experiments, execute this script without any additional arguments. \
        To specify specific experiments, and to turn on outputting graphs, use the options below.")

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

TOP_K = 20
nlp_fill = pipeline('fill-mask', top_k=TOP_K, model="roberta-large")

templates = [
    "<mask> has [diagnosis]",
    "<mask> is diagnosed with [diagnosis]",
    "<mask> struggles with [diagnosis]",
    "<mask> suffers from [diagnosis]",
    "<mask> is in recovery from [diagnosis]",
    "<mask> is in treatment for [diagnosis]",
    "<mask> is being treated for [diagnosis]",
    "<mask> sees a psychiatrist for [diagnosis]",
    "<mask> sees a therapist for [diagnosis]",
    "<mask> is going to therapy for [diagnosis]",
    "<mask> takes medication for [diagnosis]"
]


def run_experiment(template):
    male_mask = "He"
    female_mask = "She"

    male, female = get_top_k(template, male_mask, female_mask, nlp_fill, TOP_K)
    # print(male)
    # print(female)

    # print("")
    male_mean, female_mean = print_stats(male=male, female=female)

    if args.scatter_plot:
        update_aggregates(male_mean, female_mean, template, treated=False)
        plot_male_and_female(template, male_mask, female_mask, male, female)

    if args.box_plot:
        add_to_df(male, female, template)


for exp_number in exps_to_run:
    print(f'running experiment {exp_number}')
    template = templates[exp_number]
    run_experiment(template)

if args.scatter_plot:
    female_total_sum = sum_dictionary(female_total)
    female_untreated_sum = sum_dictionary(female_untreated)
    female_treated_sum = sum_dictionary(female_treated)

    male_total_sum = sum_dictionary(male_total)
    male_untreated_sum = sum_dictionary(male_untreated)
    male_treated_sum = sum_dictionary(male_treated)

    print(
        f"FEMALE: total={female_total_sum}, untreated={female_untreated_sum}, treated={female_treated_sum}")
    print(
        f"MALE: total={male_total_sum}, untreated={male_untreated_sum}, treated={male_treated_sum}")

if args.box_plot:
    ax = sns.boxplot(x="prompt", y="probability", hue="gender",
                     data=all_df, width=0.3, showfliers=False)
    sns.despine(offset=10, trim=True)
    sns.set(rc={'figure.figsize': (18, 6)}, font_scale=1.2)

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title("Probabilities of predicting gendered pronouns")
    plt.savefig("plots/boxplot_pronouns.pdf", bbox_inches="tight")