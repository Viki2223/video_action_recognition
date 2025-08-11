import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    Single-object Kalman filter tracker for bounding boxes.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with initial bounding box.
        bbox: [x1, y1, x2, y2] in absolute pixel coordinates
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement function
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Measurement noise
        self.kf.R[2:,2:] *= 10.0
        
        # Process noise
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Covariance matrix
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """
        Update the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        
    def predict(self):
        """
        Advance the state vector and return predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
        
    def get_state(self):
        """
        Return the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)
        
    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Convert bounding box [x1,y1,x2,y2] to z format [x,y,s,r].
        x,y is the center of the box
        s is the scale/area
        r is the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        s = w * h  # scale/area
        r = w / float(h)  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
        
    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Convert state x to bounding box [x1,y1,x2,y2].
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2.0, x[1]-h/2.0, x[0]+w/2.0, x[1]+h/2.0]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2.0, x[1]-h/2.0, x[0]+w/2.0, x[1]+h/2.0, score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assign detections to tracked objects using IoU.
    
    Returns:
        matched: array of matched detection/tracker pairs
        unmatched_dets: array of unmatched detections
        unmatched_trks: array of unmatched trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
            
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
    else:
        matched_indices = np.empty(shape=(0, 2))
        
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            
    # Filter out matched with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def iou(bb_test, bb_gt):
    """
    Compute IoU between two bounding boxes in [x1,y1,x2,y2] format.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


class DeepSORT:
    """
    DeepSORT tracker implementation.
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        max_age: Maximum number of frames to keep alive a track without associated detections
        min_hits: Minimum number of associated detections before track is initialized
        iou_threshold: Minimum IoU for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections, features=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: list of detections in format [x1,y1,x2,y2,score]
            features: optional features for appearance matching (not used in this implementation)
            
        Returns:
            list of active tracks in format [x1,y1,x2,y2,id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to existing trackers
        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)
            
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
            
        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))