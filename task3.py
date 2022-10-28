import numpy as np
import re
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def D5():
	'''
	Function: apply TF-IDF model and save it
	Include:
	1.read the task2.pkl
	2.read the task1.csv (TF)
	3.read the candidate-passages-top1000.tsv (extracting pid and passage)
	4.read the test-queries.tsv (extracting qid and query)
	4.remove the stopwords
	6.construct the word-based TF-IDF
	7.construct the sentence-based TF-IDF vector
	8.calculate the cosine_similarity
	9.output the result
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



	#6.construct the word-based TF-IDF
	TF_IDF_Word = {}
	for word,TF in tqdm(word_TF):
		if word in Inverted_Index_Dict.keys():
			TF_IDF_Word[word] = TF * 100000 * np.log(top1000_passages.shape[0]/(1 + len(Inverted_Index_Dict[word])))
		else:
			TF_IDF_Word[word] = TF * 100000 * np.log(top1000_passages.shape[0])



	#7.construct the sentence-based TF-IDF vector
	TF_IDF_Passage = []
	TF_IDF_Query = []

	#for passages
	for index,page in enumerate(tqdm(top1000_passages)):
		passage = page[-1].split() #obtain the passage content, split with words

		#for the passage in top1000_passages, calculate their TFIDF value for each words
		passage_TFIDF = {word:0 for word in passage}

		for word in passage:
			if word in TF_IDF_Word.keys():
				passage_TFIDF[word] += TF_IDF_Word[word]
			else:
				passage_TFIDF[word] += 0
		
		TF_IDF_Passage.append([page[0], passage_TFIDF])

	#for queries

	for index,page in enumerate(tqdm(test_queries)):
		query = page[-1].split() #obtain the passage content, split with words

		#for the query in test_queries, calculate their TFIDF value for each words
		query_TFIDF = {word:0 for word in query}

		for word in query:
			if word in TF_IDF_Word.keys():
				query_TFIDF[word] += TF_IDF_Word[word]
			else:
				query_TFIDF[word] += 0
		
		TF_IDF_Query.append([page[0], query_TFIDF])


	#8.calculate the cosine similarity and select top 100 for each query
	TF_IDF_Sentence_Query = []
	threshold = 0.0
	for index, (qid, query_TFIDF) in enumerate(tqdm(TF_IDF_Query)):

		candidate_result = []
		
		for idx, (pid, passage_TFIDF) in enumerate(TF_IDF_Passage):
			
			#Initially, it can calculate the denominator
			denominator = np.sqrt(np.sum(np.square(list(passage_TFIDF.values()))))*np.sqrt(np.sum(np.square(list(query_TFIDF.values()))))
			
			if denominator == 0:
				continue

			#Then, find the common words and calculate the dot product of them
			common_words = set(passage_TFIDF.keys()).intersection(query_TFIDF.keys())
			
			if len(common_words) == 0:
				continue

			dot_product = 0
			for word in common_words:
				dot_product += passage_TFIDF[word] * query_TFIDF[word]

			#add the cosine similarity to the list
			candidate_result.append([qid,pid,dot_product/denominator])

		
		#arrange the candidate result
		candidate_result = sorted(candidate_result,key=lambda x:x[-1],reverse=True)[:100]
		candidate_result = np.array(candidate_result).T
		candidate_result = candidate_result[:,candidate_result[-1,:] >= threshold]
		
		#add to output table
		TF_IDF_Sentence_Query += candidate_result.T.tolist()


	#9.output the result
	pd.DataFrame(TF_IDF_Sentence_Query,columns=['qid','pid','score']).to_csv("tfidf.csv",index=False)



def D6():
	'''
	Function: apply BM25 model and save it
	Include:
	1.read the task2.pkl
	2.read the task1.csv (TF)
	3.read the candidate-passages-top1000.tsv (extracting pid and passage)
	4.read the test-queries.tsv (extracting qid and query)
	5.remove the stopwords
	6.construct the sentence-based BM25 similarity
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


	#6.construct the sentence-based BM25 similarity
	k1 = 1.2
	k2 = 100
	b = 0.75
	N = top1000_passages.shape[0] #number of document
	L_avg = np.mean([len(i[-1].split()) for i in top1000_passages]) #average length of the document
	BM25_Sentence_Query = []
	threshold = 0.0

	for index, (qid, query) in enumerate(tqdm(test_queries)):

		candidate_result = []
		query = query.split() #obtain the query content, split with the words

		for idx, (pid, passage) in enumerate(top1000_passages):

			passage = passage.split() #obtain the passage content, split with words

			score = []
			for word in query:
				#obtain the W_i
				if word in Inverted_Index_Dict.keys():
					dfi = np.log(N/(1 + len(Inverted_Index_Dict[word])))
				else:
					dfi = np.log(N)
				W_i = np.log(( N - dfi + 0.5)/(dfi + 0.5))

				#obtain the K
				K = k1*(1-b+b*len(passage)/L_avg)

				#obtain the S_i (I don't know why it occurs an error)
				try:
					TFtd = re.findall(word," ".join(passage))
				except:
					TFtd = []
				
				if not TFtd:
					TFtd = 0
				else:
					TFtd = len(TFtd)
				S_i = (k1+1)*TFtd/(K + TFtd)

				#obtain the S2_i
				try:
					TFtq = re.findall(word," ".join(query))
				except:
					TFtq = []

				if not TFtq:
					TFtq = 0
				else:
					TFtq = len(TFtq)
				S2_i = (k2 + 1)*TFtq/(k2 + TFtq)

				#obtain the score
				score.append(W_i*S_i*S2_i)

			#calculate the similarity
			score = sum(score)
			candidate_result.append([qid, pid, score])


		#arrange the candidate result
		candidate_result = sorted(candidate_result,key=lambda x:x[-1],reverse=True)[:100]
		candidate_result = np.array(candidate_result).T
		candidate_result = candidate_result[:,candidate_result[-1,:] >= threshold]
		
		#add to output table
		BM25_Sentence_Query += candidate_result.T.tolist()


	#7.output the result
	pd.DataFrame(BM25_Sentence_Query,columns=['qid','pid','score']).to_csv("bm25.csv", index=False)


if __name__ == '__main__':
	D5()
	D6()