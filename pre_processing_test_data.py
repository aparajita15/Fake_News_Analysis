####### Pre-processing


import os
import swifter
import numpy as np
import pandas as pd
import pdb
import argparse
import pickle
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import tqdm

#####################################
'SPACY'
import spacy
#spacy.load('en')
from spacy import displacy
#import en_core_web_sm
nlp = spacy.load('en_core_web_sm')
######################################
'GLOVE VECTORS'
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
glove_file = datapath('test_glove.txt')
tmp_file = get_tmpfile("test_word2vec.txt")
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
######################################

global fake
global real 
global all_indices

def combine(ll):
	st=''
	for i in ll['entities']:
		st+=i+' '
	return st

def get_entities(text):
	## Analyzing the performance with two entity extraction packages

	###################Stanford CoreNLP Implementation
	## Stanford CoreNLP server should be running in the background for this to work
	## Extracting the article body and returning the list of words which were found as entities
	#text = row['articleBody']
	'''with open('inp.txt', 'w') as f:
		f.write(text)
	os.system("java -cp \"*\" -Xmx5g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -file inp.txt -outputFormat json")
	with open('inp.txt.json', 'r') as f1:
		js = json.load(f1)
		'''
	################ Spacy implementation
	#doc = nlp(unicode(text, "utf-8"))
	doc = nlp(text)
	t = [""]*len(doc.ents)
	i=0
	for X in doc.ents:
		t[i] = X.text
		i+=1
	return t


def get_all_labels(row ):
	reals  = ['mediabiasfactcheck', 'alexa', 'vargo']
	fakes =['mediabiasfactcheck', 'politifact', 'zimdars', 'min2', 'dailydot']
	t = []
	for i in reals:
		for j in fakes:
			label = i + '_' + j
			t+= [row[label]]
	return t

def get_labels(row ):
				global fake
				global real
				if row[fake]=='0':
					return 0
				elif row[real] == '1':
					return 1
				else:
					## The source hasn't listed this domain as either fake or real
					return -1


def main():	
	global real
	global fake
	global all_indices
	all_indices = []

	# Reading the input file
	#file_name = 'temporal_sample_with_content.csv'	 # --> #Original file
	file_name = 'temporal_testing_sample_with_content.csv' ## --> Testing data
	file_data = "C:/Users/Aparajita/Desktop/FakeNews/"+file_name;
	df = pd.read_csv(file_data, sep="\t", header=None, dtype=str)

	#Renaming the column names:
	## Changing names for consistency
	col_names = list(df.iloc[0])
	col_names[col_names.index('content')] = 'articleBody' 
	col_names[col_names.index('title')] = 'Headline'
	df.columns = col_names
	df=df.iloc[1:]

	# Label to consider
	reals  = ['mediabiasfactcheck', 'alexa', 'vargo']
	fakes =['mediabiasfactcheck', 'politifact', 'zimdars', 'min2', 'dailydot']
		
	label_names = []
	for i in reals:
		for j in fakes:
			label = i + '_' + j
			label_names += [label]

			real = i
			fake = j
			# Getting goroundtruth labels
			df[label] = df.swifter.apply( get_labels, axis=1)
			df[label] = pd.to_numeric(df[label], errors='ignore')
			df = df[df[label] >= 0]

	df['all_indices'] = np.nan
	df['all_indices'] =  df.swifter.apply( get_all_labels, axis=1) #df.columns.get_loc(df.columns[-1])
	#Labels stored by considering all the fake and real categories



	##############################################################
	'''  Processing begins here '''
	##############################################################
	#Running a simple bag of words model on the entire dataset
	#Extracting entities and then running the same model
	#Evaluaitng if the GloVe vecotrs can be useful? --> Checking how many words are out of vocabulary
	# Runnning a basic BiLSTM model (check - what features were captured <iff good accuracy and auc is obtained>)
	##############################################################
	#
	##Extracting entities
	#df['entities_head'] = np.nan
	#df['entities'] = np.nan
	#for i in range(len(df)):
	#		print(i)
	#		try:
#
#	#			df['entities'].iloc[i] = get_entities(df['articleBody'].iloc[i])
#	#			df['entities_head'].iloc[i] = get_entities(df['Headline'].iloc[i])
#	#		except:
#	#			pdb.set_trace()
#	#df['entities'] = df.swifter.apply( combine, axis=1)
#	#df['entities_head'] = df.swifter.apply( combine, axis=1)
	#
	################################################################################################################
	
	#### Loading vectorizers from the trained data00
	with open('Pickled_data/bow_tfreq_tfidf', 'rb') as fil:
		bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = pickle.load(fil)

	lim_unigram = 2000
	#test_head = df['entities_head'].tolist()
	#test_bodies = df['entities'].tolist()
	#test_label = df['all_indices'].tolist()
	#
	test_head = df['Headline'].tolist()
	test_bodies = df['articleBody'].tolist()
	test_label = df['all_indices'].tolist()

	test_set = []
	test_stances = []
	id_ref	= {}
	actual_labels =[]
	for i in range(len(test_bodies)):
				head = test_head[i]
				body_id = test_bodies[i]
				head_bow = bow_vectorizer.transform([head]).toarray()
				head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
				head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)

				body_bow = bow_vectorizer.transform([body_id]).toarray()
				body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
				body_tfidf = tfidf_vectorizer.transform([body_id]).toarray().reshape(1, -1)

				tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
				feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
				test_set.append(feat_vec)
				actual_labels.append(test_label[i])

	pdb.set_trace()
	with open('Pickled_data/testing_file_everything_without', 'wb') as wr:
		pickle.dump([test_set, actual_labels, label_names], wr)
	
	
	
if __name__ == '__main__':
				main()

