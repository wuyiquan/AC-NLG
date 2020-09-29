import os
from bert_score import BERTScorer
scorer = BERTScorer(lang="ch", batch_size=1, device='cuda:0')

logdir = './logs/S2S/decode_val_600maxenc_4beam_35mindec_150maxdec_ckpt-62256'
decodeddir = logdir + '/decoded'
referencedir = logdir + '/reference'
dir_or_files = os.listdir(decodeddir)
dir_or_files = sorted(dir_or_files)
count = 0
for file in dir_or_files:
	f = open(os.path.join(decodeddir, file), 'r', encoding='utf-8')
	decodetext = []
	for line in f.readlines():
		decodetext.append(line[1:])
	f.close()
	f = open(os.path.join(referencedir, file[0:6]+'_reference.txt'), 'r', encoding='utf-8')
	reftext = []
	for line in f.readlines():
		reftext.append(line[1:])
		# reftext.append(line[1:])
	f.close()
	# count += 1
	# if count == 10:
	# 	break
print(scorer.score(decodetext,reftext))
