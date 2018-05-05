import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from os import listdir
from os.path import join,exists
from tqdm import tnrange


%matplotlib inline

o_path='raw/tracks_id_2_uri.json'
if exists(o_path):
    with open(o_path) as f:
        tracks_id_2_uri=json.load(f)
else:
    i_path='raw/mpd/data'
    slices=listdir(i_path)
    tracks_uri=set([''])

    for i in tnrange(len(slices)):
        s=slices[i]
        with open(join(i_path,s)) as f:
            plist=json.load(f)
            for p in plist['playlists']:
                for t in p['tracks']:
                    if t['track_uri'] not in tracks_uri:
                        tracks_uri.add(t['track_uri'])

    tracks_uri.remove('')
    tracks_id_2_uri={i:t for i,t in enumerate(sorted(list(tracks_uri)))}
    with open(o_path,'w') as f:
        json.dump(tracks_id_2_uri,f)

ss=['team_info,main,acodersop,acodersop@gmail.com']
results=np.load('output/random.npy')
for i in tnrange(10000):
    s=str(results[i][0])+','
    for j in range(1,501):
        s+=tracks_id_2_uri[results[i][j]]+','
    ss.append(s[:-1])
s='\n'.join(ss)
with open('output/random.csv','w') as f:
    f.write(s)