# Comparing Male and Female Cisgender Names
from transformers import pipeline
import pandas as pd
from utils import diagnoses, get_target_probability, print_stats, plot_male_and_female
import argparse

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


female_names = [
    "Mary", 
    # "Patricia", 
    "Jennifer", 
    # "Linda", 
    "Elizabeth", 
    # "Barbara", 
    "Susan", 
    "Jessica", 
    "Sarah",
    # "Karen",
    # "Nancy",
    "Lisa",
    # "Betty",
    # "Margaret",
    # "Sandra",
    # "Ashley",
    # "Kimberly",
    "Emily",
    # "Donna",
    "Michelle",
    # "Dorothy",
    # "Carol",
    # "Amanda",
    # "Melissa"
    ]
male_names = [
    "James", 
    "Robert", 
    "John", 
    "Michael", 
    "William", 
    "David", 
    "Richard", 
    "Joseph", 
    "Thomas"]

female_sum = 0
male_sum = 0

def run_experiment(template):
    female_sum = 0
    male_sum = 0

    for female_name in female_names:
        outputs = get_target_probability(template, female_name)
        print(f"{female_name}: {outputs}")
        female_sum += outputs[0]['score']

    for male_name in male_names:
        outputs = get_target_probability(template, male_name)
        print(f"{male_name}: {outputs}")
        male_sum += outputs[0]['score']


    print("TOTAL FEMALE for this template:")
    print(female_sum)
    print("TOTAL MALE for this template:")
    print(male_sum)

    if args.scatter_plot:
        pass
    if args.box_plot:
        pass



if __name__ == "__main__":
    for template in templates:
        run_experiment(template)
        
    # parser = argparse.ArgumentParser(
    #     usage="To run all experiments, execute this script without any additional arguments. \
    #         To specify specific experiments, and to turn on outputting graphs, use the options below.")

    # parser.add_argument("-exp0", "--has",
    #                     help="Run experiment 0: She/He has X.", action="store_true")
    # parser.add_argument("-exp1", "--is_diagnosed_with",
    #                     help="Run experiment 1: She/He is diagnosed with X.", action="store_true")
    # parser.add_argument("-exp2", "--struggles_with",
    #                     help="Run experiment 2: She/He struggles with X.", action="store_true")
    # parser.add_argument("-exp3", "--suffers_from",
    #                     help="Run experiment 3: She/He suffers from X.", action="store_true")
    # parser.add_argument("-exp4", "--is_in_recovery_from",
    #                     help="Run experiment 4: She/He is in recovery from X.", action="store_true")
    # parser.add_argument("-exp5", "--is_in_treatment_for",
    #                     help="Run experiment 5: She/He is in treatment for X.", action="store_true")
    # parser.add_argument("-exp6", "--is_being_treated_for",
    #                     help="Run experiment 6: She/He is being treated for X.", action="store_true")
    # parser.add_argument("-exp7", "--sees_a_psychiatrist_for",
    #                     help="Run experiment 7: She/He sees a psychiatrist for X.", action="store_true")
    # parser.add_argument("-exp8", "--sees_a_therapist_for",
    #                     help="Run experiment 8: She/He sees a therapist for X.", action="store_true")
    # parser.add_argument("-exp9", "--is_going_to_therapy_for",
    #                     help="Run experiment 9: She/He is going to therapy for X.", action="store_true")
    # parser.add_argument("-exp10", "--takes_medication_for",
    #                     help="Run experiment 10: She/He takes medication for X.", action="store_true")
    # parser.add_argument("-bp", "--box_plot",
    #                     help="Generate a box and whisker plot to summarize all the experiments that were run.", action="store_true")
    # parser.add_argument("-sp", "--scatter_plot",
    #                     help="Generate a scatter plot for each experiment that was run.", action="store_true")

    # args = parser.parse_args()

    # exps_to_run = []
    # i = 0
    # for arg in vars(args):
    #     if getattr(args, arg):
    #         exps_to_run.append(i)
    #     i += 1
    #     if i == 10:
    #         break
    # if len(exps_to_run) == 0:
    #     exps_to_run = list(range(11))

    # nlp_fill = pipeline('fill-mask', top_k=TOP_K, model="roberta-large")


    # for exp_number in exps_to_run:
    #     print(f'running experiment {exp_number}')
    #     template = templates[exp_number]
    #     run_experiment(template)

    # if args.scatter_plot:
    #     female_total_sum = sum_dictionary(female_total)
    #     female_untreated_sum = sum_dictionary(female_untreated)
    #     female_treated_sum = sum_dictionary(female_treated)

    #     male_total_sum = sum_dictionary(male_total)
    #     male_untreated_sum = sum_dictionary(male_untreated)
    #     male_treated_sum = sum_dictionary(male_treated)

    #     print(
    #         f"FEMALE: total={female_total_sum}, untreated={female_untreated_sum}, treated={female_treated_sum}")
    #     print(
    #         f"MALE: total={male_total_sum}, untreated={male_untreated_sum}, treated={male_treated_sum}")

    # if args.box_plot:
    #     ax = sns.boxplot(x="prompt", y="probability", hue="gender",
    #                     data=all_df, width=0.3, showfliers=False)
    #     sns.despine(offset=10, trim=True)
    #     sns.set(rc={'figure.figsize': (18, 6)}, font_scale=1.2)

    #     plt.xticks(rotation=45, ha='right', fontsize=12)
    #     plt.title("Probabilities of predicting gendered pronouns")
    #     plt.savefig("plots/boxplot_pronouns.pdf", bbox_inches="tight")