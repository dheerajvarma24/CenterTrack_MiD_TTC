import json
import numpy as np

#file_path = './train.json'

def convert_json(file_path):
    # red the json file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ann = []
    count = 0
    for item in data['annotations']:
        # read velocity data
        velocity = item['velocity']
        # convert velocity to nan
        if np.isnan(velocity[0]):
            velocity = ['nan' for i in range(4)]
        
        item['velocity'] = velocity
        ann.append(item)
    
    data['annotations'] = ann

    # write the json file
    with open('converted_train.json', 'w') as f:
        json.dump(data, f)


def convert_eval(filepath):
    # read the json file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(type(data))
    print('----------')
    temp = {}
    for key, values in data.items():
        if values is not None:
            temp1 = []
            for item in values:
                item['dep_ratio'] = 'nan'
                temp1.append(item)
            temp[key]= temp1
        else:
            temp[key] = item
    
    # write the json file
    with open('converted_save_results_nuscenes.json', 'w') as f:
        json.dump(temp, f)



if __name__ == '__main__':
    # convert_json(file_path)
    convert_eval('./save_results_nuscenes.json')
