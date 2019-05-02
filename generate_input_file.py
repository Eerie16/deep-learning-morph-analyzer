import os
import string
# TRAINING_FOLDER = 'news_articles_and_heritage/Development'
TRAINING_FOLDER = 'Testing'

prev = 'asd'
# 1 for i , 2 for o, 0 for b
files = [f for f in os.listdir(TRAINING_FOLDER) if os.path.isfile(os.path.join(TRAINING_FOLDER, f))]
BASE_DIR = os.path.join(os.path.dirname(__file__),TRAINING_FOLDER)
for f in files:
	file = open(os.path.join(BASE_DIR, f), 'r')
	for line in file.readlines():
		dic={}
		orig_inp = line.split('\t')
		if(len(orig_inp)<2):
			# print()
			continue
		inp={x.split('-')[0] : x.split('-')[1] for x in orig_inp[5].split('|')}
		for key in inp:
			if inp[key]=='':
				inp[key]='Unk'
		if orig_inp[1]=='।':
			prev=inp['chunkId']
			inp['chunkId']='O-'+inp['chunkId']
		else:	
			if inp['chunkId']==prev:
				prev = inp['chunkId']
				inp['chunkId']='I-'+inp['chunkId']
			else:
				prev=inp['chunkId']
				inp['chunkId']='B-'+inp['chunkId']
		inp['chunkId']=inp['chunkId'].rstrip(string.digits)
		print(orig_inp[3], inp['gen'],inp['num'],inp['pers'],inp['case'],inp['tam'], inp['chunkId'],sep='\t')
	file.close()
