import os
import glob
import csv
import json
import pandas as pd

#nontargeted_path="/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/results/**/**/**/**/eps_2_255/json/nontargeted_perf.json"
#targeted_path="/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/results/**/**/**/**/eps_2_255/json/targeted_perf.json"
nontargeted_path="/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/results/imagenet1k/convnext_tiny/CosPGD_alpha/**/eps_2_255/**/json/nontargeted_perf.json"
targeted_path="/home/prasse/Shashank_Projects/adversarial-attacks-pytorch/results/imagenet1k/convnext_tiny/CosPGD_alpha/**/eps_2_255/**/json/targeted_perf.json"

nontargeted = glob.glob(nontargeted_path)
targeted = glob.glob(targeted_path)

df_nontargeted = None
df_targeted = None

#import ipdb;ipdb.set_trace()

for file1 in nontargeted:
    #f1 = open(file1)
    #js = js.load(f1)

    df = pd.read_json(file1, orient='index').T
    if df_nontargeted.__class__ == None.__class__:
        df_nontargeted = df
    else:
        df_nontargeted = pd.concat([df_nontargeted, df])

for file2 in targeted:
    df = pd.read_json(file2, orient='index').T
    if df_targeted.__class__ == None.__class__:
        df_targeted = df
    else:
        df_targeted = pd.concat([df_targeted, df])

df_targeted.sort_values(['dataset', 'Model', 'iterations', 'Robust_Acc'], ascending=[True, True, True, False]).to_csv('alpha_imagenet1k_targeted.csv', index=False)
df_nontargeted.sort_values(['dataset', 'Model', 'iterations', 'Robust_Acc'], ascending=[True, True, True, False]).to_csv('alpha_imagenet1k_nontargeted.csv', index=False)