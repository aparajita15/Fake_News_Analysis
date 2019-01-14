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
	## Needed because bow vectorizer tokenizes the text and doesnt accept tokens
	## Needed only when working with entitites
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
				if row[fake]=='0': 	return 0
				elif row[real] == '1': return 1
				else:
					## The source hasn't listed this domain as either fake or real
					return -1


def main():	
	global real
	global fake
	global all_indices
	all_indices = []

	# Reading the input file
	file_name = 'temporal_sample_with_content.csv'	 # --> #Original file
	#file_name = 'temporal_testing_sample_with_content.csv' ## --> Testing data
	file_data = "C:/Users/Aparajita/Desktop/FakeNews/"+file_name;
	df = pd.read_csv(file_data, sep="\t", header=None, dtype=str)

	#Renaming the column names: Changing names for consistency
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
	#Evaluaitng if the GloVe vecotrs can be useful? --> Checking how many words are out of vocabulary --> not informative
	# Runnning a basic BiLSTM model (check - what features were captured <iff good accuracy and auc is obtained>) -- complete before feb
	##############################################################
	
	#Extracting entities
	df['entities_head'] = np.nan
	df['entities'] = np.nan
	for i in range(len(df)):
		print(i)
		df['entities'].iloc[i] = get_entities(df['articleBody'].iloc[i])
		df['entities_head'].iloc[i] = get_entities(df['Headline'].iloc[i])
	df['entities'] = df.swifter.apply( combine, axis=1)
	df['entities_head'] = df.swifter.apply( combine, axis=1)
	
	################################################################################################################
	#### Extracting vectorizers from the training dataset
	stop_words = [
					"a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
					"already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
					"any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
					"became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
					"below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
					"con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
					"either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
					"everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
					"former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
					"has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
					"him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
					"into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
					"many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
					"must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
					"of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
					"ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
					"serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
					"somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
					"ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
					"therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
					"three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
					"twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
					"whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
					"wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
					"with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
					]
	lim_unigram = 2000
	train_head = df['entities_head'].tolist()
	train_bodies = df['entities'].tolist()
	train_label = df['all_indices'].tolist()

	# Using the tf and tfidf features
	bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)#, stop_words=stop_words)
	bow = bow_vectorizer.fit_transform( train_head	+ train_bodies )  # Train set only

	tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
	tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

	tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
			fit((train_head+ train_bodies) )  # T

	## Calculating the vectorizers and the training\tetsing features
	# training features

	train_set = []
	train_stances = []
	id_ref	= {}
	for i, elem in enumerate(train_head + train_bodies ):
				id_ref[elem] = i


	for i in range(len(train_head)):
					head_tfidf = tfidf_vectorizer.transform([train_head[i]]).toarray()
					body_tfidf = tfidf_vectorizer.transform([train_bodies[i]]).toarray()
					#cosine similarity
					tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
					#tf
					head_tf = tfreq[id_ref[train_head[i]]].reshape(1, -1)
					body_tf = tfreq[id_ref[train_bodies[i]]].reshape(1, -1)

					feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
					train_set.append(feat_vec)
					train_stances.append(train_label[i])

	with open('Pickled_data/training_file_everything', 'wb') as wr:
		pickle.dump([train_set, train_stances, label_names], wr)
	with open('Pickled_data/bow_tfreq_tfidf', 'wb') as fil:
		pickle.dump([bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer], fil)
	
	
if __name__ == '__main__':
				main()

