import numpy as np
import re
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import pandas as pd

def D1():
	'''
	Function: provide the normalized term frequency by zipfs law
	Includes:
	1.reading the text
	2.preprocessing: lower the text, delete the punctuation
	3.obtian the term
	4.normalized
	5.plot
	'''
	#1.reading the text
	with open("passage-collection.txt","r",encoding="utf8") as f:
		document = f.read()\
		.replace("<B>","")\
		.replace("</B>","")\
		.replace("</U>","")\
		.replace("<U>","")\
		.replace("\\","")\
		.replace("[","")\
		.replace("]","") #.splitlines()
	
	#2.preprocessing
	document = document.lower()
	symbol = "[_.!+-=——,$%^,，。?、~@#￥%……&*《》<>「」{}【】()（）/'']"
	document = re.sub(symbol,"",document)

	#3.obtian the term
	term_dict = {}
	document = document.split()
	for word in tqdm(document):
		if word in term_dict.keys():
			term_dict[word] += 1
		else:
			term_dict[word] = 1

	#4.arrange and normalization
	N_I = (1/np.array(list(term_dict.values()))).sum()
	term_dict = sorted(term_dict.items(),key=lambda x:x[1], reverse=True)
	save_term_dict = []
	for index, word in enumerate(term_dict):
		term_dict[index] = [word[0]+" "+str(index+1),float(1/(index+1)/N_I)]
		save_term_dict.append([word[0],float(1/(index+1)/N_I)])

	#5.plot
	#5.1 plot 1: the rank with the TF
	num_sample = 50
	sample_word = dict(term_dict[:num_sample][::-1])
	plt.barh(list(sample_word.keys()),list(sample_word.values()))
	plt.xlabel("The Normalized Term Frequency")
	plt.title("The Zipf's Law-Based Normalized Term Frequency")
	plt.show()

	#5.2 the log-log plot
	ranking = np.array([i+1 for i in range(num_sample)])
	zipf_law = np.array(list(dict(term_dict[:num_sample]).values()))/ranking
	plt.loglog(list(sample_word.values())[::-1],zipf_law,marker='o',markersize=7)
	plt.xlabel("The Predict Log Distribution")
	plt.ylabel("The Zipf's Law Log Distribution")
	plt.title("Log-Log Plot Between the Formula-Based Distribution with the Zipf's Law Distribution")
	plt.show()

	#6.save CSV with columns = word, TF
	df = pd.DataFrame(save_term_dict,columns=['word','TF']).to_csv("task1.csv",index=False)


if __name__ == '__main__':
	D1()