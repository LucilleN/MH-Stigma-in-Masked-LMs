from LIWCutil import *
import csv, sys

CONNO_FILES = {
  "agency": "connotation/agency_verb.csv",
  "authority": "connotation/authority_verb.csv",
}

RES_FILES = {
	"narration": "connotation_on_narration.csv",
	"dialogue": "connotation_on_dialogue.csv"
}

RESOURCE_FILES = {
	"narration": "parsed_scripts/narration.txt",
	"dialogue": "parsed_scripts/dialogue.txt"
}

""" Parses the narration data set """
def narration():
	""" NARRATION:
		m_id --> s_id --> n_id --> narr
			 --> s_id --> n_id --> ...
			 		  --> n_id --> ... 
	"""	
	with open(RESOURCE_FILES["narration"], 'r') as tsvin:
		lines = csv.reader(tsvin, delimiter='\t')
		dct = {}
		for l in lines:
			m_id = int(l[0])
			s_id = int(l[1])
			n_id = l[2].strip()
			narr = l[3].strip()
			if m_id not in dct:
				dct[m_id] = {}
			if s_id not in dct[m_id]:
				dct[m_id][s_id] = {}
			dct[m_id][s_id][n_id] = narr
		return dct

""" Parses the dialogue dataset """
def dialogue():
	""" DIALOGUE:
		m_id --> s_id --> d_id -->  'speaker' = speaker
									'listener' = listener
									'diag' = dialogue
			 --> s_id --> d_id --> ...
			 		  --> d_id
	"""	
	with open(RESOURCE_FILES["dialogue"], 'r') as tsvin:
		lines = csv.reader(tsvin, delimiter='\t')
		dct = {}
		for l in lines:
			m_id = int(l[0])
			s_id = int(l[1])
			d_id = l[2].strip()
			speaker = l[3].strip()
			listener = l[4].strip()
			diag = l[5].strip()

			if m_id not in dct:
				dct[m_id] = {}
			if s_id not in dct[m_id]:
				dct[m_id][s_id] = {}
			dct[m_id][s_id][d_id] = {'speaker': speaker, 'listener': listener, 'diag': diag}
		return dct

""" Parses the connotation frame data set into a dictionary of verb --> category """
def connotation_to_dict(fname):
	dct = {}
	with open(fname, 'r') as f:
		lines = f.readlines()[1:]
		for l in lines:
			verb, cat = l.strip().split(",")[0:2]
			if (cat == "+"):
				cat = "pos"
			elif (cat == "-"):
				cat = "neg"
			elif (cat == "="):
				cat = "ntrl" 
			dct[verb] = cat
		return dct

""" Parses connotation frame data set to a similar LIWC format used in LIWCutil """
def connotation_to_liwc_format():
	words_to_cats = {}
	parse_connotation_to_regex(CONNO_FILES["agency"], words_to_cats)
	parse_connotation_to_regex(CONNO_FILES["authority"], words_to_cats)
	return words_to_cats

""" Parses each connotation frame to regex representation """
def parse_connotation_to_regex(fname, words_to_cats):
	f = open(fname)
	lines = f.readlines()[1:]
	for l in lines:
		verb, cat = l.strip().split(",")[0:2]
		v_cpy = verb
		verb = verb.split()[0]

		if (verb[-1] == 's'):
			verb = verb[:-1] + "*"

		if (verb not in words_to_cats):
			words_to_cats[verb] = []

		words_to_cats[verb].append(v_cpy) 


""" Global dictionaries """
CF_LIWC = connotation_to_liwc_format()
AGENCY_DCT = connotation_to_dict(CONNO_FILES["agency"])
AUTHORITY_DCT = connotation_to_dict(CONNO_FILES["authority"])		

""" Does a run through all the movie scripts' narrations """
def run_for_narration():
	n = narration()

	count = 0
	total_raw = {}

	for m_id in n:
		print(m_id)
		txt = ""
		for s_id in n[m_id]:
			print("\t" + str(s_id))
			for n_id in n[m_id][s_id]:
				nt = n[m_id][s_id][n_id]
				if txt == "":
					txt = nt
				else:
					txt += ". " + str(nt)

		print("\t\tExtracting data")
		raw, percent, n_words = extract(CF_LIWC, txt)	
		count += n_words
		for r in raw:
			if r not in total_raw:
				total_raw[r] = 0
			total_raw[r] += raw[r]				

	total_percent = {k: v/count for k,v in total_raw.items()}

	return total_raw, total_percent

""" Does the run for dialogues """
def run_for_dialogue():
	d = dialogue()
	count = 0
	total_raw = {}

	for m_id in d:
		print(m_id)
		txt = ""
		for s_id in d[m_id]:
			print("\t" + str(s_id))
			for d_id in d[m_id][s_id]:
				nt = d[m_id][s_id][d_id]['diag']
				if txt == "":
					txt = nt
				else:
					txt += ". " + str(nt)

		print("\t\tExtracting data")
		raw, percent, n_words = extract(CF_LIWC, txt)	

		count += n_words
		for r in raw:
			if r not in total_raw:
				total_raw[r] = 0
			total_raw[r] += raw[r]				

	total_percent = {k: v/count for k,v in total_raw.items()}

	return total_raw, total_percent

""" Write result as a CSV """
def write_result(total_raw, total_percent, fname):
	# Prepare Data
	print("\t\tPreparing data")
	data = []
	for v in total_raw:
		a_cat = ""
		if v in AGENCY_DCT:
			a_cat = AGENCY_DCT[v]
		p_cat = ""
		if v in AUTHORITY_DCT:
			p_cat = AUTHORITY_DCT[v]

		row = [v, total_raw[v], total_percent[v], a_cat, p_cat]	
		data.append(row)

	# Write data
	print("\t\tWritting to CSV")
	with open(fname, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(data)
	return	

def main():
	n_raw, n_percent = run_for_narration()
	d_raw, d_percent = run_for_dialogue()

	write_result(n_raw, n_percent, RES_FILES["narration"])
	write_result(d_raw, d_percent, RES_FILES["dialogue"])
	return

if __name__ == "__main__":
	main()