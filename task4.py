import numpy as np
import re
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def D8():
	'''
	Function: Apply Laplacian-based smooth method to implement MLE
	Include:
	1.read the task2.pkl
	2.read the task1.csv (TF)
	3.read the candidate-passages-top1000.tsv
	4.read the test-queries.tsv (extracting qid and query)
	5.remove the stopwords
	6.Estimate the probability
	7.output the result
	'''

	#1.read the task2.pkl
	with open("task2.pkl","rb") as f: Inverted_Index_Dict = pickle.load(f)


	#2.read the task1.csv (TF)
	word_TF = pd.read_csv("task1.csv")


	#3.read the candidate-passages-top1000.tsv (extracting pid and passage)
	top1000_passages = pd.read_csv("candidate-passages-top1000.tsv",sep='\t',header=None).values[:,[1,-1]]


	#4.read the test-queries.tsv (extracting qid and query)
	test_queries = pd.read_csv("test-queries.tsv",sep='\t',header=None).values


	#5.remove the stopwords
	stopwords = word_TF.values[:50,0] #highest TF
	pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

	#for passages
	for index,page in enumerate(tqdm(top1000_passages)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		top1000_passages[index][-1] = pattern.sub('', each)

	#for queries
	for index,page in enumerate(tqdm(test_queries)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		test_queries[index][-1] = pattern.sub('', each)

	word_TF = word_TF.values[50:]



	#6.Estimate the probability
	Sentence_Query_MLE = []

	for index, (qid, query) in enumerate(tqdm(test_queries)):

		candidate_result = []
		query = query.split() #obtain the query content, split with the words

		for idx, (pid, passage) in enumerate(top1000_passages):

			passage = passage.split() #obtain the passage content, split with words

			P = 1
			for word in query:
				try:
					TFtd = re.findall(word," ".join(passage))
				except:
					TFtd = []
				if not TFtd:
					TFtd = 1 #Laplacian smooth
				else:
					TFtd = len(TFtd) + 1 #Laplacian smooth
				P *= TFtd/(len(passage) + 1) #probability MLE
			candidate_result.append([qid, pid, np.log(P)])

		#arrange the candidate result
		candidate_result = sorted(candidate_result,key=lambda x:x[-1],reverse=True)[:100]

		#add to output table
		Sentence_Query_MLE += candidate_result

	#7.output the result
	pd.DataFrame(Sentence_Query_MLE,columns=['qid','pid','score']).to_csv("laplace.csv",index=False)


def D9():
	'''
	Function: Apply lidstone-based smooth method to implement MLE
	Include:
	1.read the task2.pkl
	2.read the task1.csv (TF)
	3.read the candidate-passages-top1000.tsv
	4.read the test-queries.tsv (extracting qid and query)
	5.remove the stopwords
	6.Estimate the probability
	7.output the result
	'''

	#1.read the task2.pkl
	with open("task2.pkl","rb") as f: Inverted_Index_Dict = pickle.load(f)


	#2.read the task1.csv (TF)
	word_TF = pd.read_csv("task1.csv")


	#3.read the candidate-passages-top1000.tsv (extracting pid and passage)
	top1000_passages = pd.read_csv("candidate-passages-top1000.tsv",sep='\t',header=None).values[:,[1,-1]]


	#4.read the test-queries.tsv (extracting qid and query)
	test_queries = pd.read_csv("test-queries.tsv",sep='\t',header=None).values


	#5.remove the stopwords
	stopwords = word_TF.values[:50,0] #highest TF
	pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

	#for passages
	for index,page in enumerate(tqdm(top1000_passages)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		top1000_passages[index][-1] = pattern.sub('', each)

	#for queries
	for index,page in enumerate(tqdm(test_queries)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		test_queries[index][-1] = pattern.sub('', each)

	word_TF = word_TF.values[50:]



	#6.Estimate the probability
	Sentence_Query_MLE = []

	for index, (qid, query) in enumerate(tqdm(test_queries)):

		candidate_result = []
		query = query.split() #obtain the query content, split with the words

		for idx, (pid, passage) in enumerate(top1000_passages):

			passage = passage.split() #obtain the passage content, split with words
			
			P = 1
			epsilon = 0.1
			for word in query:
				try:
					TFtd = re.findall(word," ".join(passage))
				except:
					TFtd = []
				if not TFtd:
					TFtd = epsilon #lidstone smooth
				else:
					TFtd = len(TFtd) + epsilon #lidstone smooth
				P *= TFtd/(len(passage) + epsilon) #probability MLE and lidstone smooth
			candidate_result.append([qid, pid, np.log(P)])


		#arrange the candidate result
		candidate_result = sorted(candidate_result,key=lambda x:x[-1],reverse=True)[:100]

		#add to output table
		Sentence_Query_MLE += candidate_result

	#7.output the result
	pd.DataFrame(Sentence_Query_MLE,columns=['qid','pid','score']).to_csv("lidstone.csv",index=False)



def D10():
	'''
	Function: Apply Dirichlet-based smooth method to implement MLE
	Include:
	1.read the task2.pkl
	2.read the task1.csv (TF)
	3.read the candidate-passages-top1000.tsv
	4.read the test-queries.tsv (extracting qid and query)
	5.remove the stopwords
	6.Estimate the probability
	7.output the result
	'''

	#1.read the task2.pkl
	with open("task2.pkl","rb") as f: Inverted_Index_Dict = pickle.load(f)


	#2.read the task1.csv (TF)
	word_TF = pd.read_csv("task1.csv")

	#3.read the candidate-passages-top1000.tsv (extracting pid and passage)
	top1000_passages = pd.read_csv("candidate-passages-top1000.tsv",sep='\t',header=None).values[:,[1,-1]]


	#4.read the test-queries.tsv (extracting qid and query)
	test_queries = pd.read_csv("test-queries.tsv",sep='\t',header=None).values


	#5.remove the stopwords
	stopwords = word_TF.values[:50,0] #highest TF
	pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

	#for passages
	for index,page in enumerate(tqdm(top1000_passages)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		top1000_passages[index][-1] = pattern.sub('', each)

	#for queries
	for index,page in enumerate(tqdm(test_queries)):
		each = page[-1] #obtain the passage content
		each = each.lower()
		test_queries[index][-1] = pattern.sub('', each)

	word_TF = word_TF.values[50:]


	#6.Estimate the probability
	Sentence_Query_MLE = []
	m = 50 #mu value
	word_TF = dict(word_TF)

	for index, (qid, query) in enumerate(tqdm(test_queries)):

		candidate_result = []
		query = query.split() #obtain the query content, split with the words

		for idx, (pid, passage) in enumerate(top1000_passages):

			passage = passage.split() #obtain the passage content, split with words
			
			P = 1
			for word in query:
				#obtain the D(w)
				try:
					Dw = re.findall(word, " ".join(passage))
				except:
					Dw = []
				if not Dw:
					Dw = 0
				else:
					Dw = len(Dw)

				#obtain the D
				D = len(passage)

				#obtain the P(w|c)
				if word in word_TF.keys():
					Pwc = word_TF[word] * 100000 #re-normalization
				else:
					Pwc = 0

				#obtain the probability
				P *= (Dw + m*Pwc)/(D + m)#probability MLE
			if P == 0:
				P += 0.000000000001
			candidate_result.append([qid, pid, np.log(P)])

		#arrange the candidate result
		candidate_result = sorted(candidate_result,key=lambda x:x[-1],reverse=True)[:100]

		#add to output table
		Sentence_Query_MLE += candidate_result

	#7.output the result
	pd.DataFrame(Sentence_Query_MLE,columns=['qid','pid','score']).to_csv("dirichlet.csv",index=False)

if __name__ == '__main__':
	#D8()
	D9()
	D10()