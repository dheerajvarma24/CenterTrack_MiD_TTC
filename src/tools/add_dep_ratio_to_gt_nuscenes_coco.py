import json
import numpy as np
from collections import defaultdict


file_path = './nuscenes_val_dep.json'
# write a functoin to read the json file and loop through all the items in the json file.
def add_depth_ratio_to_nuscenes_coco(file_path):
    # read the json file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # create a dictionary to store the depth ratio
    frames_per_video_count = defaultdict(int)
    # loop through all the items in the json file
    for item in data['images']:
        # read the video id data
        video_id = item['video_id']
        frames_per_video_count[video_id] += 1

    # frames_per_video_count = {1: 234, 2: 246, 3: 246, 4: 246, 5: 240, 6: 240})
    # frames_list = [234, 480, 726, 972, 1212, 1452]
    frames_list = get_frames_list(frames_per_video_count)
    video_start_frame = [1]
    for i in range(len(frames_list)-1):
        video_start_frame.append(frames_list[i]+1)

    annotations_per_video, ann_count = gen_num_ann_per_video(data, frames_list)
    
    #print(frames_list)
    print(video_start_frame)
    #print(annotations_per_video.items())
    #print(get_cumsum(list(annotations_per_video.values())))
    #print(ann_count)
    
    return

    # reading the input data
    input_data = {} 
    for item in data['annotations']:
        image_id = item['image_id']
        track_id = item['track_id']
        depth = item['depth']
        if image_id not in input_data:
            input_data[image_id] = [(track_id, depth)]
        else:
            input_data[image_id].append((track_id, depth))
    
    # adding the depth ratio
    output_dict = {}
    for frameid in input_data.keys():
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
            if (int(frameid)-1) not in input_data.keys():
                    continue

            # Loop through all the objects in the previous frame.
            for prev_trackid, prev_depth in input_data[(int(frameid)-1)]:
                # If the object is present in the previous frame, then calculate the depth ratio.
                if trackid == prev_trackid:
                    depth_ratio = round(float(depth) / float(prev_depth), 6)
                    key = str(frameid) + '_' + str(trackid)
                    output_dict[key] = str(depth_ratio)
                    continue

    
    # adding the depth ratio to the json file
    ann = []
    for item in data['annotations']:
        frameid = item['image_id']
        trackid = item['track_id']
        key = str(frameid) + '_' + str(trackid)
        if key in output_dict:
            item['dep_ratio'] = output_dict[key]
        ann.append(item)

    # writing the json file
    data['annotations'] = ann
    with open('nuscenes_train_dep.json', 'w') as f:
        json.dump(data, f)
        


def gen_num_ann_per_video(data, frames_list):
    annotations_per_video = defaultdict(int)
    index = 0
    ann_count = 0
    for item in data['annotations']:
        ann_count += 1
        frame_id = item['image_id']
        if frame_id > frames_list[index]:
            index += 1
        annotations_per_video[index] += 1
    return annotations_per_video, ann_count    

# get cummulative sum 
def get_cumsum(values):
    cumsum = []
    cumsum.append(values[0])
    for i in range(1, len(values)):
        cumsum.append(values[i] + cumsum[i-1])
    return cumsum

def get_frames_list(frames_per_video_count):
    frames_list = []    
    for i, num_frames in enumerate(frames_per_video_count.values()):
        if i == 0:
            frames_list.append(num_frames)
        else:
            frames_list.append(num_frames + frames_list[i-1])
    return frames_list

if __name__ == '__main__':
    add_depth_ratio_to_nuscenes_coco(file_path)
    