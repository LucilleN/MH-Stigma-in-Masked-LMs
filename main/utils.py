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
import math
import argparse

diagnoses = [
    "depression",
    "anxiety",
    "OCD",
    "schizophrenia",
    "PTSD",
    "bipolar disorder",
    "anorexia",
    "bulimia",
    "psychosis",
    "panic disorder",
    "borderline personality disorder"]

diagnoses_non_mh = [
    "heart disease", 
    "cancer", 
    "stroke", 
    "respiratory disease", 
    "injuries", 
    "diabetes", 
    "Alzheimer's disease", 
    "influenza", 
    "pneumonia", 
    "kidney disease", 
    "septicemia"
]

diagnoses = diagnoses_non_mh

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

def get_top_k(template, nlp_fill, top_k):

    output_for_all_diagnoses = []

    for diagnosis in diagnoses:
        prompt = template.replace("[diagnosis]", diagnosis)
        output_list = nlp_fill(prompt, top_k=top_k)
        # output_tokens = output_list[0]['token_str']
        output_for_all_diagnoses.append([output_list])

    return output_for_all_diagnoses


def get_male_female_from_top_k(template, male_mask, female_mask, nlp_fill, top_k):
    male = []
    female = []

    for diagnosis in diagnoses:
        prompt = template.replace("[diagnosis]", diagnosis)
        output_list = nlp_fill(prompt)

        female_exist = 0
        male_exist = 0
        # print(diagnosis)
        # print(output_list)
        for i in range(top_k):

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


def get_target_probability(template, target_token, nlp_fill):
    outputs = []

    for diagnosis in diagnoses:
        prompt = template.replace("[diagnosis]", diagnosis)
        output_list = nlp_fill(prompt, targets=target_token)
        # score = output_list[0]['score']
        # scores_by_diagnosis.append(score)
        outputs.append(output_list[0])

    return outputs

def get_pseudo_log_likelihood(template, target_token, nlp_fill):
    # template = "[target] is diagnosed with [diagnosis]"
    # target_token = "He"

    pseudo_lls_for_all_diagnoses = []

    for diagnosis in diagnoses:
        template_with_diagnosis = template.replace("[diagnosis]", diagnosis)  # [target] is diagnosed with depression
        pseudo_ll_current_sentence = 0

        tokens = template_with_diagnosis.split()  # ["[target]", "is", "diagnosed", "with", "depression"]
        for index, token in enumerate(tokens):
            if token == "[target]":
                print(f"Index {index} is the [target] token that we don't want to mask; skipping")
                continue
            prompt = tokens.copy()
            prompt[index] = "<mask>"
            prompt = " ".join(prompt).replace("[target]", target_token)
            
            output_list = nlp_fill(prompt, targets=" " + token)
            score = output_list[0]['score']
            pseudo_ll_current_sentence += math.log(score)

            print(f"prompt: {prompt}")
            print(f"pseudo_ll_current_sentence: {pseudo_ll_current_sentence}")

        # scores_by_diagnosis.append(score)
        # outputs.append(output_list[0])
        print(f"FINAL SUMMED pseudo_ll_current_sentence: {pseudo_ll_current_sentence}")
        pseudo_lls_for_all_diagnoses.append(pseudo_ll_current_sentence)

    print(f"returning: {pseudo_lls_for_all_diagnoses}")
    return pseudo_lls_for_all_diagnoses

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
    plt.savefig(f"plots/scatterplots/{filename}.pdf", bbox_inches="tight")


def sum_dictionary(template_to_mean_dict):
    sum = 0
    for template, mean in template_to_mean_dict.items():
        sum += mean
    return sum


def read_data_from_file(filepath):
    data = []
    with open(filepath) as f:
        # reader = csv.reader(f)
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line.strip())
    return data


def get_gender_name_scores_for_top_k(template, male_names, female_names, nlp_fill, top_k):
    male = []
    female = []

    for diagnosis in diagnoses:
        prompt = template.replace("[diagnosis]", diagnosis)
        output_list = nlp_fill(prompt)

        female_exist = 0
        male_exist = 0
        # print(diagnosis)
        # print(output_list)
        for i in range(top_k):

            if output_list[i]['token_str'] in female_names:
                female.append(output_list[i]['score'])
                female_exist = 1
            if output_list[i]['token_str'] in male_names:
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

def parse_args():
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

    return args