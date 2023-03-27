import cv2, math
import numpy as np

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

l_pair = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), 
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)
]

# from openpose
p_color = [
    (100, 100, 100), (100, 0, 0), (150, 0, 0), (200, 0, 0), (255, 0, 0),
    (100, 100, 0), (150, 150, 0), (200, 200, 0), (255, 255, 0), (0, 100, 50),
    (0, 150, 75), (0, 200, 100), (0, 255, 125), (0, 50, 100), (0, 75, 150),
    (0, 100, 200), (0, 125, 255), (100, 0, 100), (150, 0, 150), (200, 0, 200),
    (255, 0, 255) 
]

line_color = [
    (100, 100, 100), (100, 0, 0), (150, 0, 0), (200, 0, 0), (255, 0, 0),
    (100, 100, 0), (150, 150, 0), (200, 200, 0), (255, 255, 0), (0, 100, 50),
    (0, 150, 75), (0, 200, 100), (0, 255, 125), (0, 50, 100), (0, 75, 150),
    (0, 100, 200), (0, 125, 255), (100, 0, 100), (150, 0, 150), (200, 0, 200),
    (255, 0, 255) 
]

def visualize_hands(frame, hands_bboxes, hands_kps, hands_kp_scores, vis_thres=0.05):
    img = frame.copy()
    for bbox, kps, kp_scores in zip(hands_bboxes, hands_kps, hands_kp_scores):
        part_line = {}
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), RED, 1)
        # Draw keypoints
        for n in range(21):
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kps[n, 0]), int(kps[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            cv2.circle(img, (int(cor_x), int(cor_y)), 5, p_color[n], thickness=-1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], thickness=2)
    return img
