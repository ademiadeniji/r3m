import pandas as pd
import numpy
import os
import os.path as path
import pdb

videos = []
rlbench_dir = '/shared/mandi/all_rlbench_data'
for lang in os.listdir(rlbench_dir):
    lang_dir = path.join(rlbench_dir, lang)
    if not path.isdir(lang_dir):
        continue
    # if len(videos) > 100: break
    print(f'processing language dir {lang_dir}')
    for variation in os.listdir(lang_dir):
        variation_dir = path.join(lang_dir, variation)
        episodes_dir = path.join(variation_dir, "episodes")
        print(f'processing variation dir {variation_dir}')
        for episode in os.listdir(episodes_dir):
            episode_dir = path.join(episodes_dir, episode)
            if not path.isdir(episode_dir):
                continue
            print(f'processing episode dir {episode_dir}')
            for view in os.listdir(episode_dir):
                view_dir = path.join(episode_dir, view)
                if not path.isdir(view_dir):
                    continue
                print(f'processing view dir {view_dir}')
                video = {'txt': lang, 'len': len(os.listdir(view_dir)), 'path': view_dir}
                videos.append(video)
                
df = pd.DataFrame(videos)
df.to_csv('/shared/ademi_adeniji/r3m/rlbench_manifest.csv')  
