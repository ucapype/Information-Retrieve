import numpy as np
import re
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def D3():
	'''
	Function: construct the Inverted index for each passage in candidate-passages-top1000.tsv
	Includes:
	1.read the task1.csv to obtain the term fequency
	2.read the candidate-passages-top1000.tsv
	3.delete the repeat passage
	4.remove the stopwords
	5.construct the Inverted index
	'''

	#1.read the task1.csv
	word_TF = pd.read_csv("task1.csv")


	#2.read the candidate-passages-top1000.tsv
	top1000_passages = pd.read_csv("candidate-passages-top1000.tsv",sep='\t')


	#3.delete the repeat passage
	top1000_passages.columns=['qid','pid','query','passage']
	top1000_passages = top1000_passages.drop_duplicates(subset=['pid','passage']).values

	#4.remove the stopwords
	stopwords = word_TF.values[:50,0] #highest TF
	pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
	
	for index,page in enumerate(tqdm(top1000_passages)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		top1000_passages[index][-1] = pattern.sub('', each)
	
	#5.construct the Inverted index (only retain the pid)
	Inverted_Index_Dict = {}
	for index,page in enumerate(tqdm(top1000_passages)):
		each = page[-1].split() #obtain the passage content, split with words
		for word in each:
			if word in Inverted_Index_Dict.keys():
				Inverted_Index_Dict[word].append(page[1]) #add passage id to the dict
			else:
				Inverted_Index_Dict[word] = [page[1]]

	with open("task2.pkl","wb") as f: pickle.dump(Inverted_Index_Dict,f) #it can only save the pkl



if __name__ == '__main__':
	D3()