import json
import math

# TODO modify this to only pedetrains and cars; class ID while readng the json file
def mid_loss_for_nuscenes(gt_path, pred_path):
    # read the json file
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    gt_data = {}
    for item in data['annotations']:
        # cat 1 = car and 6 = pedestrian
        if item['dep_ratio'] != 'nan' and str(item['category_id']) in ['1', '6']:
            frame_id = item['image_id']
            track_id = item['track_id']
            dep_ratio = float(item['dep_ratio'])
            bbox = item['bbox']
            l,t,r,b = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            confidence_score = 1
            gt_data[str(frame_id)+'_'+str(track_id)] = [(l,t,r,b), dep_ratio, confidence_score]
    
    # read the json file
    with open(pred_path, 'r') as f:
        data = json.load(f)
    
    pred_data = {}
    for key, values in data.items():
        if values is not None:
            for item in values:
                if item['dep_ratio'] != 'nan' and str(item['class']) in ['1', '6']:
                    frame_id = key
                    track_id = item['tracking_id']
                    dep_ratio = float(item['dep_ratio'])            
                    bbox = item['bbox']
                    l,t,r,b = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    confidence_score = item['score']
                    pred_data[str(frame_id)+'_'+str(track_id)] = [(l,t,r,b), dep_ratio, confidence_score]

    # match the gt and pred data
    matched_data = {}
    count = 0
    for gt_key in gt_data.keys():
        gt_frameid = gt_key.split('_')[0]
        current_score = 0

        for pred_key in pred_data.keys():
            if pred_key.startswith(gt_frameid):
                    # calculate IOU between the two bounding boxes.
                    pred_box = pred_data[pred_key][0]
                    gt_box = gt_data[gt_key][0]

                    iou_score = calculate_iou(pred_box, gt_box)

                    # If the IOU is less than 0.5, then ignore those objects.
                    if iou_score < 0.5:
                        continue

                    # Store the matched data with the maximum iou score.
                    if iou_score > current_score:
                        '''if count == 8:
                            print('iou_score: ', iou_score)
                            print('pred_data[pred_key]: ', pred_data[pred_key])
                            print('gt_data[gt_key]: ', gt_data[gt_key])
                        count += 1'''
                        matched_data[gt_key] = [gt_data[gt_key], pred_data[pred_key]]
                        current_score = iou_score
                    elif iou_score == current_score:
                        # if IOU score is same, then check for the maximum detection condidence score for the object.
                        if pred_data[pred_key][2] > matched_data[gt_key][1][2]:
                            matched_data[gt_key] = [gt_data[gt_key], pred_data[pred_key]]
                            current_score = iou_score
    
    # calculate the loss
    losses = []
    matched_dep_ratio_gt_pred = []
    for matched_key in matched_data.keys():
        gt_dep_ratio = matched_data[matched_key][0][1]
        pred_dep_ratio = matched_data[matched_key][1][1]

        matched_dep_ratio_gt_pred.append((gt_dep_ratio, pred_dep_ratio))

        # Ignore the nan values in gt and pred
        if str(gt_dep_ratio).lower() == 'nan' or str(pred_dep_ratio).lower() == 'nan':
            continue
        loss = abs(math.log(pred_dep_ratio) - math.log(float(gt_dep_ratio)))
        losses.append(loss)
    
    # calculate the mean loss
    if len(losses) == 0:
        mean_loss = 0
    else:
        mean_loss = sum(losses)/len(losses) * 10000
    
    print('mean loss: ', mean_loss)
    print('number of matched data: ', len(matched_data))

    for i in range(0, 25):
        print('gt: ', matched_dep_ratio_gt_pred[i][0], ' pred: ', matched_dep_ratio_gt_pred[i][1])
    

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

if __name__ == '__main__':
    mid_loss_for_nuscenes('./nuscenes_val_dep.json','./converted_save_results_nuscenes_dep_ratio.json')

    #print(calculate_iou([250,500,400,550], [5,200,400,880]))