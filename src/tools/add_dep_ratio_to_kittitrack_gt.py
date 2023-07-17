import numpy as np
import fileinput
import argparse


class GTDepRatio(object):
    def __init__(self, gt_path):
        self.index = 0
        self.gt_path = gt_path
        self.input_dict = {}
        self.output_dict = {}

    def readinput(self):
        with open(self.gt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                row = line.split(' ')
                frameid = row[0]
                trackid = row[1]
                # depth = row[-2] # row -2 is the depth info in gt
                depth = row[-4] # row -4 is the depth info in pred.
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
            if frameid == str(0):
                for trackid, depth in self.input_dict[frameid]:
                    key = frameid + '_' + trackid
                    self.output_dict[key] = np.nan
                continue

            # Loop through all the objects in the current frame.
            for trackid, depth in self.input_dict[frameid]:
                
                # Ignore the Dontcare objects in the gt
                if trackid == str(-1):
                    key = frameid + '_' + trackid
                    self.output_dict[key] = np.nan
                    continue
                
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
            # Iterate over each line in the file
            for line in file:
                # Modify the line as desired
                new_line = line.strip()
                row = new_line.split(' ')
                key = row[0] + '_' + row[1]
                if key in self.output_dict.keys():
                    # Append the depth ratio to the end of the line.
                    row.append(self.output_dict[key])
                    new_line = ' '.join([str(i) for i in row]) + '\n'
                else:
                    new_line = line + '\n'
                # Write the modified line back to the file.
                # NOTE: Here the print statement is explicitly writing the content to the file.
                print(new_line, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter gt and pred validation file paths')

    # Add the required arguments
    parser.add_argument('--gt_path', type=str, help='Enter gt val file path')
    # Parse the arguments
    args = parser.parse_args()

    #for i in range (0, 21):
    gt_val_file_path = args.gt_path
    for i in [4,11]:
        if i < 10:
            filename = gt_val_file_path+'000'+str(i)+'.txt'
        else:
            filename = gt_val_file_path+'00'+str(i)+'.txt'
        gtDepRatio = GTDepRatio(filename)
        gtDepRatio.readinput()
        gtDepRatio.add_depth_ratio()
        gtDepRatio.inplace_edit()
        print('processed file: ', filename)

