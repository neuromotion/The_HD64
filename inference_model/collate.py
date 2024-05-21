import json,os,glob
from pathlib import Path

BASE_PATH = '/media/data_cifs/projects/prj_deepspine/hd64/ckpts/StimToEMG/'
EXP_PATH = 'HD64RetestSession1/*.json'

files = []
#for path in Path(os.path.join(BASE_PATH, EXP_PATH)).rglob('.json'):                                                   
for path in glob.glob(os.path.join(BASE_PATH, EXP_PATH)):
    files.append(path)

output_filename = 'HD64RetestSession1.json'
if os.path.exists(output_filename):
    os.remove(output_filename)

cjson = open(output_filename, 'a', encoding='utf-8')

collated_json = []
for idx, f in enumerate(files):
    X = json.load(open(f,'r'))
    #collated_json.update({idx: X})
    collated_json.append(X)
    
json.dump(collated_json, cjson, ensure_ascii=False, indent='\t')
