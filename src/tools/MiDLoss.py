# Script to calculate motion in depth loss.
# The corresponding predicted objects are matched to gt objects based on the minimum distance (greedy) between the bounding boxes (ltrb values).
import math

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
                type = row[2] # represents 'Car', 'Pedestrian', 'Cyclist'
                l = float(row[6])
                t = float(row[7])
                r = float(row[8])
                b = float(row[9])
                key = frameid + '_' + trackid

                if filetype == 'gt':
                    dep_ratio = row[17] # there are nan values in gt, so let it be string not float(row[17])
                    self.gt_data[key] = [(l,t,r,b), dep_ratio, type]
                elif filetype == 'pred':
                    dep_ratio = float(row[-1])
                    self.pred_data[key] = [(l,t,r,b), dep_ratio, type]


    # Match the pred data with gt data based on the minimum distance between the bounding boxes (ltrb values).
    # To minimise the computation, we can match the pred data with gt data based on the frameid.
    def match_pred_with_gt(self):
        for pred_key in self.pred_data.keys():
            # Key is in the format of 'frameid_trackid'
            pred_frameid = pred_key.split('_')[0]
            # Set matching score to Max allowed value.
            score = 1e+6

            for gt_key in self.gt_data.keys():
                # Get all the gt values with the same frameid as of the current pred frameid
                if gt_key.startswith(pred_frameid):
                    # calculate matching score
                    pred_box = self.pred_data[pred_key][0]
                    gt_box = self.gt_data[gt_key][0]
                    current_score = abs(pred_box[0] - gt_box[0]) + abs(pred_box[1] - gt_box[1]) + abs(pred_box[2] - gt_box[2]) + abs(pred_box[3] - gt_box[3])

                    # Store the matched data with the minimum score.
                    if current_score < score:
                        self.matched_data[pred_key] = [self.pred_data[pred_key], self.gt_data[gt_key]]
                        score = current_score

    # Read the dep_ratio info from the matched_data obtained from above and calculate the loss.
    def cal_MID_Loss(self):
        # Loop through all the matched data
        for matched_key in self.matched_data.keys():
            pred_dep_ratio = self.matched_data[matched_key][0][1]
            gt_dep_ratio = self.matched_data[matched_key][1][1]
            pred_type = self.matched_data[matched_key][0][2]

            # Ignore the nan values and cyclist type.
            if str(gt_dep_ratio).lower() == 'nan' or str(pred_dep_ratio).lower() == 'nan' or str(pred_type).lower() == 'cyclist':
                continue
            loss = abs(math.log(pred_dep_ratio) - math.log(float(gt_dep_ratio)))
            self.losses.append(loss)


if __name__ == '__main__':
    final_loss = []
    for i in range(0,21):
        if i < 10:
            gt_path = '../../data/kitti_tracking/label_02_val_half/000' + str(i) + '.txt'
            pred_path = '../../exp/tracking,ddd/kitti_half/results_kitti_tracking/000' + str(i) + '.txt'
        else:
            gt_path = '../../data/kitti_tracking/label_02_val_half/00' + str(i) + '.txt'
            pred_path = '../../exp/tracking,ddd/kitti_half/results_kitti_tracking/00' + str(i) + '.txt'

        calMIDLoss = CalMIDLoss(gt_path, pred_path)
        calMIDLoss.read_data_from_files(gt_path, 'gt')
        calMIDLoss.read_data_from_files(pred_path, 'pred')
        calMIDLoss.match_pred_with_gt()
        calMIDLoss.cal_MID_Loss()
        final_loss.append( sum(calMIDLoss.losses)/len(calMIDLoss.losses)  * 10000)
        print(f'file name {pred_path}, loss {final_loss[i]:.4f}')
        print(f'file name {pred_path}, len of the file {len(calMIDLoss.loss)}')

    print(f'Average loss {sum(final_loss)/len(final_loss):.4f}')
