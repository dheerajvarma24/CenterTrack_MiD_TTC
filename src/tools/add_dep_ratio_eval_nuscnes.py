import json
import numpy as np
from collections import defaultdict

file_path = './converted_save_results_nuscenes.json'

def add_depth_ratio_to_nuscenes_coco(file_path):
    # read the json file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    #print(frames_list)
    video_start_frame = [1, 241, 481, 721, 961, 1201, 1441, 1681, 1921, 2161, 2401, 2641, 2881, 3121, 3355, 3595, 3829, 4063, 4297, 4531, 4765, 5005, 5245, 5485, 5725, 5959, 6193, 6433, 6673, 6907, 7147, 7387, 7621, 7855, 8095, 8335, 8569, 8803, 9037, 9277, 9517, 9757, 9997, 10237, 10471, 10705, 10939, 11179, 11419, 11653, 11893, 12139, 12385, 12625, 12865, 13105, 13351, 13597, 13843, 14089, 14329, 14569, 14809, 15055, 15301, 15541, 15781, 16021, 16261, 16501, 16741, 16981, 17221, 17467, 17707, 17953, 18193, 18433, 18673, 18913, 19153, 19393, 19639, 19879, 20125, 20371, 20611, 20851, 21097, 21337, 21583, 21829, 22075, 22315, 22555, 22795, 23035, 23275, 23515, 23755, 24001, 24247, 24487, 24733, 24973, 25219, 25459, 25699, 25939, 26179, 26419, 26665, 26911, 27157, 27403, 27649, 27895, 28141, 28387, 28627, 28867, 29107, 29353, 29593, 29833, 30073, 30313, 30553, 30799, 31039, 31285, 31531, 31777, 32017, 32257, 32503, 32743, 32983, 33229, 33469, 33709, 33949, 34189, 34429, 34669, 34909, 35149, 35395, 35635, 35875]

    # print(video_start_frame)
    #print(annotations_per_video.items())
    #print(get_cumsum(list(annotations_per_video.values())))
    #print(ann_count)
    

    # reading the input data
    input_data = {} 
    for key, values in data.items():
        if values is not None:
            for item in values:
                image_id = key
                track_id = item['tracking_id']
                depth = item['dep'][0]
                

                if image_id not in input_data:
                    input_data[image_id] = [(track_id, depth)]
                else:
                    input_data[image_id].append((track_id, depth))
        else:
            input_data[key] = None
    
    # print(type(list(input_data.keys())[0]))
    
    # adding the depth ratio
    output_dict = {}
    count = 0
    for frameid in input_data.keys():
        
        if input_data[frameid] is None:
            key = str(frameid) + '_' + str(-1)
            output_dict[key] = []
            continue

        # objects in the first frame have no depth ratio
        if frameid in video_start_frame:
        
            for trackid, depth in input_data[frameid]:
                key = str(frameid) + '_' + str(trackid)
                output_dict[key] = 'nan'
            continue
        
        # Loop through all the objects in the current frame.
        for trackid, depth in input_data[frameid]:
            
            
            # first assign all objects in the current frame to nan
            key = str(frameid) + '_' + str(trackid)
            output_dict[key] = 'nan'

            # todo handle if missing frames like 65
            if (str(int(frameid)-1)) not in input_data.keys():
                    continue

            # Loop through all the objects in the previous frame.
            for prev_trackid, prev_depth in input_data[str(int(frameid)-1)]:
                
                # If the object is present in the previous frame, then calculate the depth ratio.
                if trackid == prev_trackid:
                    
                        
                        
                    depth_ratio = round(float(depth) / float(prev_depth), 6)
                    key = str(frameid) + '_' + str(trackid)
                    output_dict[key] = str(depth_ratio)
                    continue

    # adding the depth ratio to the json file
    ann = {}
    for key, values in data.items():
        if values is None:
            ann[key] = values
            continue
    
        temp = []
        for item in values:
            thiskey = str(key) + '_' + str(item['tracking_id'])
            if thiskey in output_dict:
                item['dep_ratio'] = output_dict[thiskey]
            else:
                item['dep_ratio'] = 'nan'
            temp.append(item)
        
        ann[key] = temp

    # writing the json file
    data = ann
    with open('converted_save_results_nuscenes_dep_ratio.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    add_depth_ratio_to_nuscenes_coco(file_path)