from os.path import exists
import pandas as pd
import requests
import json
import os
import time

def adding_ripe_atlas_probes(YEAR,MONTH,DAY):
    if not(exists('probes/'+ YEAR+ MONTH + DAY + '.json')):
        r = requests.get('https://ftp.ripe.net/ripe/atlas/probes/archive/'+YEAR+'/'+MONTH+'/'+ YEAR+ MONTH + DAY + '.json.bz2', stream=True)
        with open('probes/' + YEAR+ MONTH + DAY + '.json.bz2', 'wb') as fd:
            for chunk in r.iter_content():
                fd.write(chunk)
        os.system("bunzip2 probes/" + YEAR+ MONTH + DAY + '.json.bz2')
    probes = {d['id']: d for d in json.load(open('probes/'+ YEAR+ MONTH + DAY + '.json'))['objects']}
    df_probes = pd.DataFrame(probes).transpose()
    return df_probes


if __name__ == "__main__":
    TODAY = time.strftime("%Y-%m-%d")
    YEAR, MONTH, DAY = TODAY.split('-')
    DAY = '01'
    adding_ripe_atlas_probes(YEAR,MONTH, DAY)