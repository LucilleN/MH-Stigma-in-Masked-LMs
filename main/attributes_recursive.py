from transformers import pipeline
# from scipy import stats
# import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
# import matplotlib.pylab as plt
# from nrclex import NRCLex
import argparse
from tqdm import tqdm, trange
from utils import *
import matplotlib.pylab as plt
import seaborn as sns


parser = argparse.ArgumentParser()


parser.add_argument("-mh", default="", type=str)
parser.add_argument("-model", default="roberta", type=str)

args = parser.parse_args()


MODEL = args.model
MH = args.mh

print (MODEL, MH)

models = {
    'roberta': {
        'huggingface_path': "roberta-large",
        'mask_token': "<mask>"
    },
    'mentalroberta': {
        'huggingface_path': "mental/mental-roberta-base",
        'mask_token': "<mask>"
    },
    'bert': {
        'huggingface_path': "bert-large-uncased",
        'mask_token': "[MASK]"
    },
    'clinicallongformer': {
        'huggingface_path': "yikuan8/Clinical-Longformer",
        'mask_token': "<mask>"
    },
    'mentalbert': {
        'huggingface_path': "mental/mental-bert-base-uncased",
        'mask_token': "[MASK]"
    },
    # 'clinicalpubmedbert': {
    #     'huggingface_path': "Tsubasaz/clinical-pubmed-bert-base-512",
    #     'mask_token': "[MASK]"
    # },
    # 'psychsearch': {
    #     'huggingface_path': "nlp4good/psych-search",
    #     'mask_token': "[MASK]"
    # }
}

MODEL_PATH = models[MODEL]['huggingface_path']

male_subjects = [
    "congressman",
    "congressmen",
    "men",
    "man",
    "he",
    "his",
    "him",
    "his",
    "mr.",
    "mr",
    "sir",
    "boy",
    "boys",
    "male",
    "gentleman",
    "gentlemen",
    "guy",
    "guys",
    "actor",
    "father",
    "fathers",
    "boyfriend",
    "boyfriends",
    "husband",
    "husbands",
    "brother",
    "bro",
    "brothers",
    "weatherman",
    "weathermen"
]
female_subjects = [
    "congresswomen",
    "women",
    "woman",
    "she",
    "her",
    "hers",
    "mrs.",
    "ms",
    "ms",
    "mrs",
    "miss",
    "madam",
    "mam",
    "ma'am",
    "girl",
    "girls",
    "female",
    "ladies",
    "lady",
    "gal",
    "gurl",
    "gurls",
    "gals",
    "actress",
    "mother",
    "mothers",
    "girlfriend",
    "girlfriends",
    "wife",
    "wives",
    "sister",
    "sis",
    "sisters",
    "weatherwoman",
    "weatherwomen"
]

female_set = set(female_subjects)
male_set = set(male_subjects)

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

templates = [
    ("I would feel aggravated by a <mask> who has [diagnosis].", "Anger"),
    ("I would feel unsafe around a <mask> who has [diagnosis].", "Dangerousness"),
    ("A <mask> with [diagnosis] would terrify me.", "Fear"),
    ("I am angry at a <mask> with [diagnosis].", "Anger"),
    ("I would require a <mask> with [diagnosis] to take medication if I were in charge of the treatment.", "Coercion"),
    ("I think a <mask> with [diagnosis] poses a risk to the neighbors unless hospitalized.", "Segregation"),
    ("If I were an employer, I would not interview a <mask> with [diagnosis].", "Avoidance"),
    ("I would be willing to talk to a <mask> about their problems with [diagnosis].", "Help"),
    ("I would feel pity for a <mask> with [diagnosis].", "Pity"),
    ("I would think that a <mask> should be responsible for their present condition of [diagnosis].", "Blame"),
    ("I think the <mask>'s present condition of [diagnosis] is controllable.", "Blame"),
    ("I feel irritated by a <mask> with [diagnosis].", "Anger"),
    ("I feel a <mask> with [diagnosis] is dangerous.", "Dangerousness"),
    ("I think a <mask> with [diagnosis] should be forced into treatment even if they do not want to.", "Coercion"),
    ("I think it would be best for the community if a <mask> with [diagnosis] were put away in a psychiatric hospital.", "Segregation"),
    ("I would share a car pool with a <mask> with [diagnosis] every day.", "Avoidance"),
    ("I think an asylum is the best place for a <mask> with [diagnosis] where they can be kept away from the neighbors.", "Segregation"),
    ("I would feel threatened by a <mask> with [diagnosis].", "Dangerousness"),
    ("I feel scared of a <mask> with [diagnosis].", "Fear"),
    ("It is likely I would help a <mask> with [diagnosis].", "Help"),
    ("I feel certain that I would help a <mask> with [diagnosis].", "Help"),
    ("I feel much sympathy for a <mask> with [diagnosis].", "Pity"),
    ("I think a <mask> with [diagnosis] is responsible for their own present condition.", "Blame"),
    ("I feel frightened of a <mask> with [diagnosis].", "Fear"),
    ("I would force a <mask> with [diagnosis] to live in a group home if I were in charge of the treatment.", "Coercion"),
    ("If I were a landlord, I probably would rent an apartment to a <mask> with [diagnosis].", "Avoidance"),
    ("I feel much concern for a <mask> with [diagnosis].", "Pity")
]

nlp_fill = pipeline('fill-mask', model=MODEL_PATH)

MASK_TOKEN = models[MODEL]['mask_token']

def get_top_k(template, nlp_fill, top_k):
    template = template.replace('<mask>', MASK_TOKEN)
    output_list = nlp_fill(template, top_k=top_k)
    return output_list

def update_template_generate(template, new_token, nlp_fill, step, beam_size=3, prob=1):
    if step == 1:
        new_template = template.replace(MASK_TOKEN, f"{MASK_TOKEN} {new_token}")
    else:
        new_template = template.replace(MASK_TOKEN, f"{new_token} {MASK_TOKEN}")
#     print (new_template)
    return new_template

def get_probability_recursive_3steps(test, nlp_fill, top_k, beam_size):
    
#     output_file_path = f'/projects/bdata/inna/stigma/MH-Stigma-in-Masked-LMs/output/attribute_recursive_3steps_TOPK_{top_k}_BEAM_{beam_size}.csv'
    
    output_print = []
    
    TOP_K = top_k
    BEAM_SIZE = beam_size
    accu_prob = 0
    female_prob = 0
    male_prob = 0
    total_valid_prompt = 0
    
#     print (get_top_k(test, nlp_fill, TOP_K))
    result = get_top_k(test, nlp_fill, TOP_K)
#     print (result)
    for token_idx in range(TOP_K):

        token = result[token_idx]['token_str'].strip()
        if (token not in male_set) and (token not in female_set):
            new_test = update_template_generate(test, token, nlp_fill, step=1)

            prob = result[token_idx]['score']
            result2 = get_top_k(new_test, nlp_fill, BEAM_SIZE)
            for token2_idx in range(BEAM_SIZE):
                token2 = result2[token2_idx]['token_str'].strip()
                if (token2 not in male_set) and (token2 not in female_set):
                    new_test2 = update_template_generate(new_test, token2, nlp_fill, step=2)

                    prob2 = result2[token2_idx]['score']
                    result3 = get_top_k(new_test2, nlp_fill, BEAM_SIZE)
                    for token3_idx in range(BEAM_SIZE):
                        token3 = result3[token3_idx]['token_str'].strip()
                        if (token3 not in male_set) and (token3 not in female_set):
#                             new_test3 = update_template_generate(new_test2, token3, nlp_fill, step=3)
                            continue
                        else:
                            prob3 = result3[token3_idx]['score']
#                             print ((token3, prob3), (token2, prob2), (token, prob))
                            
#                             print ((result3[token3_idx]['sequence'], prob*prob2*prob3), file = output_file)
                            output_print.append((result3[token3_idx]['sequence'], prob*prob2*prob3))
                            if token3 in female_set:
                                female_prob += prob * prob2 * prob3
                            elif token3 in male_set:
                                male_prob += prob * prob2 * prob3
                            accu_prob += prob * prob2 * prob3
#                             print (accu_prob)
                            total_valid_prompt += 1

                else:
                    prob2 = result2[token2_idx]['score']
#                     print ((token2, prob2), (token, prob))
#                     print ((result2[token2_idx]['sequence'], prob*prob2), file = output_file)
                    output_print.append((result2[token2_idx]['sequence'], prob*prob2))
                    if token2 in female_set:
                        female_prob += prob * prob2 
                    elif token2 in male_set:
                        male_prob += prob * prob2 
                    accu_prob += prob * prob2
#                     print (accu_prob)
                    total_valid_prompt += 1
        else:
            prob = result[token_idx]['score']
#             print ((result[token_idx]['sequence'], prob), file = output_file)
            output_print.append((result[token_idx]['sequence'], prob))
            if token in female_set:
                female_prob += prob  
            elif token in male_set:
                male_prob += prob 
            accu_prob += prob
#             print (accu_prob)
            total_valid_prompt += 1

    # print (total_valid_prompt)
    # print ("female prob: ", female_prob)
    # print ("male prob: ", male_prob)
    
#     output_df = pd.DataFrame(output_print, columns=['sequence', 'probability'])
#     output_df.to_csv(output_file_path)
#     print (output_print)
    
    return female_prob, male_prob, output_print 

female_prob_list = []
male_prob_list = []
output_log = []
label_log = []

diagnosis_list = []
sequence_list = []
category_list = []

if MH == "":
    DIAG_LIST = diagnoses
elif MH == "non":
    DIAG_LIST = diagnoses_non_mh

for diagnosis in tqdm(DIAG_LIST):
    for template_pair in templates:
        
        template = template_pair[0]
        template = template.replace("[diagnosis]", diagnosis)
#         print (template)
        female_prob, male_prob, output_seq = get_probability_recursive_3steps(template, nlp_fill, 10, 10)
        female_prob_list.append(female_prob)
        male_prob_list.append(male_prob)
        output_log.extend(output_seq)
        label_log.extend([(diagnosis, template_pair[1])]*len(output_seq))
        diagnosis_list.append(diagnosis)
        sequence_list.append(template_pair[0])
        category_list.append(template_pair[1])
        
        
label_log_df = pd.DataFrame(label_log, columns=['diagnosis','stigma_category'])
output_log_df = pd.DataFrame(output_log, columns=['sequence', 'probability'])
log_df = pd.concat([label_log_df, output_log_df], axis=1)
output_file_path = f'/projects/bdata/inna/stigma/MH-Stigma-in-Masked-LMs/output/attribute_recursive_3steps_TOPK_10_BEAM_10_{MODEL}_{MH}.csv'
log_df.to_csv(output_file_path)

gender_prob = pd.DataFrame(np.array([female_prob_list, male_prob_list, diagnosis_list, sequence_list, category_list]).T, columns=['female_prob', 'male_prob' , ' diagnosis', 'sequence', 'stigma_category'])
gender_prob.to_csv(f'/projects/bdata/inna/stigma/MH-Stigma-in-Masked-LMs/output/attribute_recursive_3steps_TOPK_10_BEAM_10_{MODEL}_{MH}_result.csv')

