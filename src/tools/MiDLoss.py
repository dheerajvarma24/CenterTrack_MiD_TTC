# Script to calculate motion in depth loss.
import math
import argparse

class CalMIDLoss:
    def __init__(self, gt_path, pred_path):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.gt_data = {}
        self.pred_data = {}
        self.matched_data = {}
        self.losses = []

    # While reading the data from the gt and pred files, store the data in a dictionary with key as 'frameid_trackid' and value as a list of [(l,t,r,b), dep_ratio, type]
    def read_data_from_files(self, path, filetype):
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                row = line.split(' ')
                frameid = row[0]
                trackid = row[1]
                classid = row[2]
                l = float(row[6])
                t = float(row[7])
                r = float(row[8])
                b = float(row[9])
                key = frameid + '_' + trackid

                if filetype == 'gt':
                    confidence_score = 1
                    dep_ratio = row[-1] # there are nan values in gt, so let it be string not float(row[17])
                    self.gt_data[key] = [(l,t,r,b), dep_ratio, confidence_score, classid]
                elif filetype == 'pred':
                    confidence_score = float(row[-2])
                    dep_ratio = float(row[-1])
                    self.pred_data[key] = [(l,t,r,b), dep_ratio, confidence_score, classid]



    #Identify all the objects in the pred frame.
    #For each object in the pred frame, find the object in the gt frame with the highest IOU.
    def match_pred_with_gt(self):
        for gt_key in self.gt_data.keys():
            # Key is in the format of 'frameid_trackid'
            gt_frameid = gt_key.split('_')[0]
            # Set matching score to Max allowed value.
            #score = 1e+6
            current_score = 0

            for pred_key in self.pred_data.keys():
                # Get all the pred values with the same frameid as of the current gt frameid and also check for the class id.
                if pred_key.startswith(gt_frameid) and self.pred_data[pred_key][-1].lower() == self.gt_data[gt_key][-1].lower(): # check for class id also 
                    # calculate IOU between the two bounding boxes.
                    pred_box = self.pred_data[pred_key][0]
                    gt_box = self.gt_data[gt_key][0]

                    iou_score = CalMIDLoss.calculate_iou(pred_box, gt_box)

                    # If the IOU is less than 0.5, then ignore those objects.
                    if iou_score < 0.5:
                        continue

                    # Store the matched data with the maximum iou score.
                    if iou_score > current_score:
                        self.matched_data[gt_key] = [self.gt_data[gt_key], self.pred_data[pred_key]]
                        current_score = iou_score
                    elif iou_score == current_score:
                        # if IOU score is same, then check for the maximum detection condidence score for the object.
                        if self.pred_data[pred_key][2] > self.matched_data[gt_key][1][2]:
                            self.matched_data[gt_key] = [self.gt_data[gt_key], self.pred_data[pred_key]]
                            current_score = iou_score
    

    def calculate_iou(pred_box, gt_box):
        # Calculate the intersection area
        # inter_area = (min(pred_box[2], gt_box[2]) - max(pred_box[0], gt_box[0])) * (min(pred_box[3], gt_box[3]) - max(pred_box[1], gt_box[1]))
        x1, y1, x2, y2 = pred_box
        x3, y3, x4, y4 = gt_box
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        # Calculate the union area
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        union_area = pred_area + gt_area - inter_area

        # Calculate the IOU
        iou_score = inter_area / union_area
        return iou_score

    # Read the dep_ratio info from the matched_data obtained from above and calculate the loss.
    def cal_MID_Loss(self):
        # Loop through all the matched data
        for matched_key in self.matched_data.keys():
            gt_dep_ratio = self.matched_data[matched_key][0][1]
            pred_dep_ratio = self.matched_data[matched_key][1][1]

            # Ignore the nan values in gt and pred
            if str(gt_dep_ratio).lower() == 'nan' or str(pred_dep_ratio).lower() == 'nan':
                continue
                
            # (optional) Ignore the objects with dep ratio > 1
            if float(gt_dep_ratio) > 1:
                continue
            
            # Multiply with 10^4 at the end for the final loss. 
            loss = abs(math.log(pred_dep_ratio) - math.log(float(gt_dep_ratio)))
            self.losses.append(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter gt and pred validation file paths')

    # Add the required arguments
    parser.add_argument('--gt_path', type=str, help='Enter gt val file path')
    parser.add_argument('--pred_path', type=str, help='Enter pred val file path')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the arguments are provided
    if not args.gt_path or not args.pred_path:
        print('provide the gt and pred validation file paths as arguments')
    
    # Read the gt and pred validation file paths
    gt_val_file_path = args.gt_path
    pred_val_file_path = args.pred_path

    final_loss = {}
    # 4 and 11 are validatoin data files.
    for i in [4,11]:
        if i < 10:
            gt_path = gt_val_file_path + '/000' + str(i) + '.txt'
            pred_path = pred_val_file_path + '/000' + str(i) + '.txt'
        else:
            gt_path = gt_val_file_path + '/00' + str(i) + '.txt'
            pred_path = pred_val_file_path + '/00' + str(i) + '.txt'

        calMIDLoss = CalMIDLoss(gt_path, pred_path)
        calMIDLoss.read_data_from_files(gt_path, 'gt')
        calMIDLoss.read_data_from_files(pred_path, 'pred')
        calMIDLoss.match_pred_with_gt()
        calMIDLoss.cal_MID_Loss()
        final_loss[i] = sum(calMIDLoss.losses)/len(calMIDLoss.losses)  * 10000
        print(f'file name {pred_path}, loss {final_loss[i]:.4f}')
        print(f'file name {pred_path}, len of the file {len(calMIDLoss.losses)}')
      
    print(f'Average MiD Loss {sum(list(final_loss.values()))/len(list(final_loss.values())):.4f}')
    
