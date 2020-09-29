from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import os

rouge = Rouge()

bleuscore1 = 0
bleuscore2 = 0
bleuscoren = 0

rougescore1 = 0
rougescore2 = 0
rougescorel = 0

ref_len = 0
gen_len = 0

logdir = './logs/AC-NLG/decode_val_300maxenc_4beam_35mindec_150maxdec_ckpt-48652'
decodeddir = logdir + '/decoded_id_new'
referencedir = logdir + '/reference_id_new'
dir_or_files = os.listdir(decodeddir)
dir_or_files = sorted(dir_or_files)
# print(dir_or_files)
blue = []
for file in dir_or_files:
	f = open(os.path.join(decodeddir, file), 'r', encoding='utf-8')
	decodetext = ''
	for line in f.readlines():
		decodetext += line
	f.close()
	f = open(os.path.join(referencedir, file[0:6]+'_reference.txt'), 'r', encoding='utf-8')
	reftext = ''
	for line in f.readlines():
		reftext += line
	
	f.close()
	ref_len += len(reftext.split())
	gen_len += len(decodetext.split())

	reference = [reftext]
	candidate = decodetext
	rouge_score = rouge.get_scores(decodetext, reftext)
	rougescore1 += rouge_score[0]["rouge-1"]['r']
	rougescore2 += rouge_score[0]["rouge-2"]['r']
	rougescorel += rouge_score[0]["rouge-l"]['r']

	bleuscore1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
	bleuscore2 += sentence_bleu(reference, candidate,weights=(0, 1, 0, 0))
	bleuscoren += sentence_bleu(reference, candidate,weights=(0.25, 0.25, 0.25, 0.25))

bleuscore1 /= len(dir_or_files)
bleuscore2 /= len(dir_or_files)
bleuscoren /= len(dir_or_files)
rougescore1 /= len(dir_or_files)
rougescore2 /= len(dir_or_files)
rougescorel /= len(dir_or_files)

ref_len /= len(dir_or_files)
gen_len /= len(dir_or_files)

print('b1%.3f'%bleuscore1)
print('b2%.3f'%bleuscore2)
print('bn%.3f'%bleuscoren)
print('r1%.3f'%rougescore1)
print('r2%.3f'%rougescore2)
print('rl%.3f'%rougescorel)
print('reflen%.3f'%ref_len)
print('genlen%.3f'%gen_len)