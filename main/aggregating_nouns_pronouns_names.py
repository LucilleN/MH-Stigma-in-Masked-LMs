# Comparing He vs She
from transformers import pipeline
import pandas as pd
from utils import diagnoses, get_target_probability, get_top_k, print_stats, plot_male_and_female, read_data_from_file
import matplotlib.pylab as plt
import seaborn as sns
import argparse

# templates = [
#     "<mask> has [diagnosis]",
#     "<mask> is diagnosed with [diagnosis]",
#     "<mask> struggles with [diagnosis]",
#     "<mask> suffers from [diagnosis]",
#     "<mask> is in recovery from [diagnosis]",
#     "<mask> is in treatment for [diagnosis]",
#     "<mask> is being treated for [diagnosis]",
#     "<mask> sees a psychiatrist for [diagnosis]",
#     "<mask> sees a therapist for [diagnosis]",
#     "<mask> is going to therapy for [diagnosis]",
#     "<mask> takes medication for [diagnosis]"
# ]
templates = [
    "[MASK] has [diagnosis]",
    "[MASK] is diagnosed with [diagnosis]",
    "[MASK] struggles with [diagnosis]",
    "[MASK] suffers from [diagnosis]",
    "[MASK] is in recovery from [diagnosis]",
    "[MASK] is in treatment for [diagnosis]",
    "[MASK] is being treated for [diagnosis]",
    "[MASK] sees a psychiatrist for [diagnosis]",
    "[MASK] sees a therapist for [diagnosis]",
    "[MASK] is going to therapy for [diagnosis]",
    "[MASK] takes medication for [diagnosis]"
]

# Aggregates for the following She/He experiments
TOP_K = 100
probability_threshold = 0.01
female_untreated, female_treated, female_total = {}, {}, {}
male_untreated,   male_treated,   male_total   = {}, {}, {}
all_df = None

male_subjects = [
    "he",
    "man",
    "men",
    "father", 
    "dad",
    "guy",
    "dude",
    "brother", 
    "son",
    "boy"
]
female_subjects = [
    "she",
    "woman",
    "women",
    "mother", 
    "mom",
    "girl",
    "gal",
    "sister", 
    "daughter",
    "lady"
]
female_names = read_data_from_file("../data/women_top_1000_names_only.csv")
male_names = read_data_from_file("../data/men_top_1000_names_only.csv")
for name in female_names:
    if name in male_names:
        female_names.remove(name)
        male_names.remove(name)

def update_aggregates(male_mean, female_mean, template, treated=False):
    female_total[template] = female_mean
    male_total[template] = male_mean

    if treated:
        female_treated[template] = female_mean
        male_treated[template] = male_mean
    else:
        female_untreated[template] = female_mean
        male_untreated[template] = male_mean


def add_to_df(male, female, template):  # , current_df=None):
    global all_df
    new_add = pd.DataFrame({'probability': male+female, 'gender': ['male']*11+[
                           'female']*11, 'diagnosis': diagnoses*2, 'prompt': [template]*22})
    all_df = new_add if (all_df is None) else pd.concat([all_df, new_add])



def run_experiment(template):
    # male_mask = "He"
    # female_mask = "She"

    print(f"TOP {TOP_K} OUTPUTS FOR THE TEMPLATE {template}")
    top_k_for_all_diagnoses = get_top_k(template, nlp_fill, TOP_K)

    male_scores = []
    female_scores = []
    
    for top_k_for_one_diagnosis in top_k_for_all_diagnoses:
        outputs = top_k_for_one_diagnosis[0]
        score_m_for_template_with_this_diagnosis = 0
        score_f_for_template_with_this_diagnosis = 0
        for output in outputs:
            score = output['score']
            if score < probability_threshold:
                break
            token_str = output['token_str']
            full_sentence = output['sequence']
            print(f"{score}, {token_str}, {full_sentence}")

            if token_str.lower() in male_subjects or token_str in male_names:
                score_m_for_template_with_this_diagnosis = score_m_for_template_with_this_diagnosis + score
            elif token_str.lower() in female_subjects or token_str in female_names:
                score_f_for_template_with_this_diagnosis = score_f_for_template_with_this_diagnosis + score
        print(f"end of finding options for one template with one diagnosis; score_m = {score_m_for_template_with_this_diagnosis}, score_f = {score_f_for_template_with_this_diagnosis}")
        male_scores.append(score_m_for_template_with_this_diagnosis)
        female_scores.append(score_f_for_template_with_this_diagnosis)


    male_mean, female_mean = print_stats(male=male_scores, female=female_scores)

    # if args.scatter_plot:
    #     update_aggregates(male_mean, female_mean, template, treated=False)
    #     plot_male_and_female(template, male_mask, female_mask, male_scores, female_scores)

    if args.box_plot:
        add_to_df(male_scores, female_scores, template)


if __name__ == "__main__":
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

    # nlp_fill = pipeline('fill-mask', top_k=TOP_K, model="roberta-large")
    # nlp_fill = pipeline('fill-mask', model="mental/mental-roberta-base")
    # nlp_fill = pipeline('fill-mask', model="emilyalsentzer/Bio_ClinicalBERT")
    # nlp_fill = pipeline('fill-mask', model="yikuan8/Clinical-Longformer")
    # nlp_fill = pipeline('fill-mask', model="Tsubasaz/clinical-pubmed-bert-base-512")
    nlp_fill = pipeline('fill-mask', model="nlp4good/psych-search")


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
        sns.despine(offset=10)
        sns.set(rc={'figure.figsize': (18, 6)}, font_scale=1.2)

        plt.xticks(rotation=45, ha='right', fontsize=12)
        ax.set_ylim([0, 0.6])
        plt.title("Probabilities of predicting gendered pronouns")
        plt.savefig(f"../plots/boxplot_aggregated_psychsearch_p{probability_threshold}.pdf", bbox_inches="tight")
        # plt.savefig("../plots/boxplot_pronouns_mentalroberta_AGG.pdf", bbox_inches="tight")
        # plt.savefig("../plots/boxplot_pronouns_clinicalbert_AGG.pdf", bbox_inches="tight")
        # plt.savefig("../plots/boxplot_pronouns_clinicallongformer_AGG.pdf", bbox_inches="tight")
        # plt.savefig("../plots/boxplot_pronouns_clinicalpubmedbert_AGG.pdf", bbox_inches="tight")
        # plt.savefig("../plots/boxplot_pronouns_psychsearch_AGG.pdf", bbox_inches="tight")