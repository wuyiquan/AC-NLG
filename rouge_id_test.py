#! /usr/bin/env python
#! -*- coding=utf-8 -*-

# import pyrouge
import os

def rouge_eval(ref_dir, dec_dir):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  #logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  return r.output_to_dict(rouge_results)


base_dir = './logs/AC-NLG/decode_val_300maxenc_4beam_35mindec_150maxdec_ckpt-48652'

word_ref_path="%s/%s"%(base_dir,"reference")
word_dec_path="%s/%s"%(base_dir,"decoded")
id_ref_path="%s/%s"%(base_dir,"reference_id_new")
id_dec_path="%s/%s"%(base_dir,"decoded_id_new")

if os.path.exists(id_ref_path):
    os.system("rm -rf %s"%(id_ref_path))
os.makedirs(id_ref_path)

if os.path.exists(id_dec_path):
    os.system("rm -rf %s"%(id_dec_path))
os.makedirs(id_dec_path)

word2id = {}

# reference
for fn in os.listdir(word_ref_path):
    word_ref_file = "%s/%s"%(word_ref_path,fn) 
    id_ref_file = "%s/%s"%(id_ref_path,fn) 
    with open(word_ref_file,'r') as in_f,open(id_ref_file,'w') as out_f:
        line = in_f.read()[1:]
        out_list = []
        for word in line.strip().split():
        # for word in line.strip().replace(' ',''):
            if word not in word2id: 
                word2id[word] = len(word2id)
            out_list += [word2id[word]]
        out_f.write(' '.join([str(_x) for _x in out_list]))

# decode 
for fn in os.listdir(word_dec_path):
    word_dec_file = "%s/%s"%(word_dec_path,fn) 
    id_dec_file = "%s/%s"%(id_dec_path,fn) 
    with open(word_dec_file,'r') as in_f,open(id_dec_file,'w') as out_f:
        line = in_f.read()[1:]
        out_list = []
        for word in line.strip().split():
        # for word in line.strip().replace(' ',''):
            if word not in word2id: 
                word2id[word] = len(word2id)
            out_list += [word2id[word]]
        out_f.write(' '.join([str(_x) for _x in out_list]))

# results_dict = rouge_eval(id_ref_path,id_dec_path)

# log_str = ""
# for x in ["1","2","l"]:
#     log_str += "\nROUGE-%s:\n" % x
#     for y in ["f_score", "recall", "precision"]:
#         key = "rouge_%s_%s" % (x,y)
#         key_cb = key + "_cb"
#         key_ce = key + "_ce"
#         val = results_dict[key]
#         val_cb = results_dict[key_cb]
#         val_ce = results_dict[key_ce]
#         log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
# print(log_str)
