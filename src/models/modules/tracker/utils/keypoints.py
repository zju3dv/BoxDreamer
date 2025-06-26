import torch

import cv2
import numpy as np

def traditional_detector(method, frames, idxs):
    # input frames should be image tensor
    # shape: B T C H W , data range: 0-1, float
    # convert tensor to numpy array and make image list
    # make rgb to gray
    device = frames.device
    batch_list = []
    for b in range(frames.shape[0]):
        frame_list = []
        for i in range(frames.shape[1]):
            frame = frames[b][i].cpu().numpy().transpose(1, 2, 0) * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame_list.append(frame)

        batch_list.append(frame_list)
    

    
    
    assert method in ['sift', 'orb']
    if method == 'sift':
        detector = cv2.SIFT_create()
    elif method == 'orb':
        detector = cv2.ORB_create()
    else:
        raise ValueError('method not supported')
    
    batch_query_points = []
    for batch in batch_list:
        query_points = []
        for idx, i in zip(idxs, batch):
            # use sift to get feature points
            kp = detector.detect(i, None)
            # convert cv2 keypoint to numpy array
            kp = np.array([[k.pt[0], k.pt[1]] for k in kp])

            # make query point tensor (B, N, 3) # 3 is t, x, y; t is frame index
            query_points.append(np.concatenate([np.array([idx]*kp.shape[0]).reshape(-1, 1), kp], axis=1))
        
        query_points = np.concatenate(query_points, axis=0)
        
        batch_query_points.append(query_points)
        
    # pad the query points to make the same length
    max_len = max([i.shape[0] for i in batch_query_points])
    for i in range(len(batch_query_points)):
        if batch_query_points[i].shape[0] < max_len:
            pad = np.zeros((max_len - batch_query_points[i].shape[0], 3))
            batch_query_points[i] = np.concatenate([batch_query_points[i], pad], axis=0)
    
    return torch.stack([torch.tensor(i).float().unsqueeze(0).to(device) for i in batch_query_points], dim=0).squeeze(1)

def neural_detector(method='superpoint', frames=[], idxs=[]):
    from transformers import AutoImageProcessor, SuperPointForKeypointDetection
    
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    
    # frames should be in Image format
    import PIL.Image as Image
    frames = [Image.fromarray(i) for i in frames]
    
    inputs = processor(images=frames, return_tensors="pt")
    outputs = model(**inputs)
    
    # handle the output and get the feature points in the format of (B, N, 3)
    query_points = []
    for idx, i in zip(idxs, range(len(frames))):
        image_mask = outputs.mask[i]
        image_indices = torch.nonzero(image_mask).squeeze()
        image_keypoints = outputs.keypoints[i][image_indices]
        image_scores = outputs.scores[i][image_indices]
        
        for keypoint, score in zip(image_keypoints, image_scores):
            keypoint_x, keypoint_y = keypoint[0].item(), keypoint[1].item()
            query_points.append([idx, keypoint_x, keypoint_y])
        
    query_points = torch.tensor(query_points).float().unsqueeze(0).cuda()
    
    return query_points
        