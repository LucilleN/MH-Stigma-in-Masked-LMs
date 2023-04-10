"""
This script runs experiments for RQ1: Do MLMs associate mental health conditions with a particular gender?

To run this script:
- First make sure your HuggingFace access token is in ACCESS_TOKEN.txt. 
- Then, run `python rq1.py` in your terminal.

All results / scores will be saved to the `output/` folder. To generate figures, see `figures.ipynb`.
"""

from transformers import pipeline
import pandas as pd
from utils import diagnoses, get_top_k, print_stats, read_data_from_file

templates = [
    # Diagnosed with condition (Phase 1 of Health Action Process Approach)
    "<mask> has [diagnosis]",
    "<mask> is diagnosed with [diagnosis]",
    "<mask> is struggling with [diagnosis]",
    "<mask> suffers from [diagnosis]",

    # Intending to seek treatment (Phase 2 of Health Action Process Approach)
    "<mask> is looking for treatment for [diagnosis]",
    "<mask> wants to see a therapist for [diagnosis]",
    "<mask> is looking for a therapist for [diagnosis]",
    "<mask> is trying to find a psychiatrist for [diagnosis]",
    "<mask> is searching for treatment for [diagnosis]",
    "<mask> wants to get help for [diagnosis]",

    # Taking action to get treatment (Phase 3 of Health Action Process Approach)
    "<mask> is in treatment for [diagnosis]",
    "<mask> is being treated for [diagnosis]",
    "<mask> sees a psychiatrist for [diagnosis]",
    "<mask> sees a therapist for [diagnosis]",
    "<mask> is in therapy for [diagnosis]",
    "<mask> takes medication for [diagnosis]",
    "<mask> is in recovery from [diagnosis]"
]

models = {
    # 'RoBERTa': {
    #     'huggingface_path': "roberta-large",
    #     'mask_token': "<mask>"
    # },
    'mentalroberta': {
        'huggingface_path': "mental/mental-roberta-base",
        'mask_token': "<mask>"
    },
    'clinicallongformer': {
        'huggingface_path': "yikuan8/Clinical-Longformer",
        'mask_token': "<mask>"
    }
}

TOP_K = 100
probability_threshold = 0.01
female_untreated, female_treated, female_total = {}, {}, {}
male_untreated,   male_treated,   male_total   = {}, {}, {}
ambig_untreated,  ambig_treated,  ambig_total  = {}, {}, {}
all_results_df = None

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
ambig_names = []
for name in female_names:
    if name in male_names:
        female_names.remove(name)
        male_names.remove(name)
        ambig_names.append(name)


def add_to_all_results_df(male, female, ambig, template): 
    n = 11
    global all_results_df
    new_add = pd.DataFrame({
        'probability': male+female+ambig, 
        'gender': ['Male']*n+['Female']*n+['Unspecified']*n, 
        'diagnosis': diagnoses*3, 
        'prompt': [template]*3*n})
    all_results_df = new_add if (all_results_df is None) else pd.concat([all_results_df, new_add])


def run_experiment(template):

    print(f"TOP {TOP_K} OUTPUTS FOR THE TEMPLATE {template}")
    top_k_for_all_diagnoses = get_top_k(template, nlp_fill, TOP_K)

    male_scores = []
    female_scores = []
    ambig_scores = []
    
    for top_k_for_one_diagnosis in top_k_for_all_diagnoses:
        outputs = top_k_for_one_diagnosis[0]
        score_m_for_template_with_this_diagnosis = 0
        score_f_for_template_with_this_diagnosis = 0
        score_a_for_template_with_this_diagnosis = 0
        for output in outputs:
            score = output['score']
            if score < probability_threshold:
                break
            token_str = output['token_str']
            full_sentence = output['sequence']
            print(f"{score} probability for {token_str} in '{full_sentence}'")

            if token_str.lower() in male_subjects or token_str in male_names:
                score_m_for_template_with_this_diagnosis = score_m_for_template_with_this_diagnosis + score
            elif token_str.lower() in female_subjects or token_str in female_names:
                score_f_for_template_with_this_diagnosis = score_f_for_template_with_this_diagnosis + score
            else:
                score_a_for_template_with_this_diagnosis = score_a_for_template_with_this_diagnosis + score

        male_scores.append(score_m_for_template_with_this_diagnosis)
        female_scores.append(score_f_for_template_with_this_diagnosis)
        ambig_scores.append(score_a_for_template_with_this_diagnosis)


    print(f"RESULTS FOR TEMPLATE: {template}")
    male_mean, female_mean = print_stats(male=male_scores, female=female_scores)

    print(f"len(male_scores): {len(male_scores)}")
    print(f"len(female_scores): {len(female_scores)}")
    print(f"len(ambig_scores): {len(ambig_scores)}")
    add_to_all_results_df(male_scores, female_scores, ambig_scores, template)


if __name__ == "__main__":

    huggingface_access_token = ""
    with open('ACCESS_TOKEN.txt', 'r') as file:
        huggingface_access_token = file.read().rstrip()

    for model in models:

        print(f"""\n\n####################\n\n  MODEL: {model}  \n\n####################\n\n""")

        nlp_fill = pipeline('fill-mask', model=models[model]['huggingface_path'], use_auth_token=huggingface_access_token)

        num_experiments = len(templates)
        for exp_number in range(num_experiments):
            print(f'running experiment {exp_number}')
            template = templates[exp_number].replace("<mask>", models[model]['mask_token'])
            run_experiment(template)

        if all_results_df is not None:
            # all_results_df.to_csv(f"../output/{model}_all_results_df_non_mh.csv")
            # all_results_df.to_csv(f"../output/{model}_all_results_df.csv")
            all_results_df.to_csv(f"../output/{model}_all_results_df_intention.csv")
            # all_results_df.to_csv(f"../output/{model}_all_results_df_intention_non_mh.csv")
        all_results_df = None