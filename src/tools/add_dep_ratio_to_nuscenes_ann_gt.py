### This is file is not used anymore; can clean this up.

# Before executing this script, please make sure that you ran 'convert_nuScenes.py' and got the train, val, ... json files.
# Then extract only the anotations from the json files, then flatten and save them to a csv file. Use online tools such as https://jsoneditoronline.org/ to perform these steps.


##### Final format should be like this:
# id,image_id,category_id,dim[0],dim[1],dim[2],location[0],location[1],location[2],depth,occluded,truncated,rotation_y,amodel_center[0],amodel_center[1],iscrowd,track_id,attributes,velocity[0],velocity[1],velocity[2],velocity[3],bbox[0],bbox[1],bbox[2],bbox[3],area,alpha

# 1,1,6,1.642,0.621,0.669,18.63882979851619,1.0145927635241274,59.02486732065484,59.02486732065484,0,0,-3.120828115749234,1216.1754150390625,495.6607666015625,0,1,4,0.00044349978428388765,-0.019996846097421125,-0.0003095560961708178,0.0,1206.5693751819117,477.86111828160216,19.31993062031279,35.78389940122628,691.342453755944,2.8564488103607584

import numpy as np
import fileinput
import csv

class GTDepRatioNuscenes(object):
    def __init__(self, gt_path):
        self.index = 0
        self.gt_path = gt_path
        self.input_dict = {}
        self.output_dict = {}

    def readinput(self):
        with open(self.gt_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                frameid = row['image_id']
                trackid = row['track_id']
                depth = row['depth']
                if frameid in self.input_dict:
                    self.input_dict[frameid].append((trackid, depth))
                else:
                    self.input_dict[frameid] = [(trackid, depth)]
        return self.input_dict

    def add_depth_ratio(self):
        self.index = 0
        # Loop through all the the frames in gt
        for frameid in self.input_dict.keys():
            # Objects in the first frame have no depth ratio. 
            if frameid == str(1):
                for trackid, depth in self.input_dict[frameid]:
                    key = frameid + '_' + trackid
                    self.output_dict[key] = np.nan
                continue

            # Loop through all the objects in the current frame.
            for trackid, depth in self.input_dict[frameid]:
                
                # First assigning all the objects with nan, later we will replace the nan with actual depth ratio.
                key = frameid + '_' + trackid
                self.output_dict[key] = np.nan

                # Few frame are missing for example:240 frame in file 5; 318 frame in file 9
                if str(int(frameid)-1) not in self.input_dict.keys():
                    continue

                # Loop through all the objects in the previous frame.
                for trackid_prev, depth_prev in self.input_dict[str(int(frameid)-1)]:
                    # If the object is present in the previous frame, then calculate the depth ratio.
                    if trackid == trackid_prev:
                        depth_ratio = round(float(depth) / float(depth_prev), 6)
                        key = frameid + '_' + trackid
                        self.output_dict[key] = str(depth_ratio)
                        continue
        
        return self.output_dict


    # This function is used to edit the groundtruth files inplace.
    def inplace_edit(self):
        # Open the file for in-place editing
        with fileinput.FileInput(self.gt_path, inplace=True) as file:
            reader = csv.DictReader(file, delimiter=',')
            # Iterate over each line in the file
            for row in reader:
                # Modify the line as desired
                #new_line = line.strip()
                #row = new_line.split(' ')
                key = row['image_id'] + '_' + row['track_id']
                #if key in self.output_dict.keys():
                    # Append the depth ratio to the end of the line.
                # row.append(self.output_dict[key])
                row['dep_ratio'] = self.output_dict[key]
                new_line = ','.join([str(i) for i in row.values()]) + '\n'
                
                # Write the modified line back to the file.
                # NOTE: Here the print statement is explicitly writing the content to the file.
                print(new_line, end='')


if __name__ == '__main__':
    train_file = '../../data/nuscenes/nuscenes_mini_train_anno.csv'
    val_file = '../../data/nuscenes/nuscenes_mini_val_anno.csv'
    for file_name in [train_file, val_file]:
        
        gtDepRatio = GTDepRatioNuscenes(file_name)
        gtDepRatio.readinput()
        gtDepRatio.add_depth_ratio()
        gtDepRatio.inplace_edit()
        print('processed file: ', file_name)


# After obtaining the annotations with depth ratio, convert the csv file to json file. Make use of online tool suchas https://jsoneditoronline.org/.