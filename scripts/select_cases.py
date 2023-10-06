import pickle
import os
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription

import logging

logger = logging.getLogger(__name__)



if __name__ == '__main__':

    with open("processed_data/dataset_summary.pkl", "rb") as f:
        summary_dict = pickle.load(f)
    
    new_summary = {}
    count = 0
    for obj_id, summary in summary_dict.items():
        if summary['track_length'] != 91:
            continue

        sdc_id = summary['sdc_id']
        objects_of_interest = summary['objects_of_interest']

        if sdc_id not in objects_of_interest:
            continue
        
        os.system(f'cp processed_data/{obj_id} raw_scenes/{count}.pkl')
        count += 1