from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="")
ap.add_argument("-c", "--class", help="name of class", default="car")
args = vars(ap.parse_args())

input_video = args["input"]
input_video_name = input_video.split('/')[-1]

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")


# id_and_se_bbox_list = {}


def main(yolo):
    start = time.time()
    # Definition of the parameters
    max_cosine_distance = 0.4  # ***
    nn_budget = None
    nms_max_overlap = 0.5  # ***

    counter = []
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    # video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(input_video)
    fps_of_video = video_capture.get(cv2.CAP_PROP_FPS)
    # print('fps_of_video:', fps_of_video)
    out = None

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_file_name = os.path.join('outputs', input_video_name + '_output.avi')
        out = cv2.VideoWriter(output_file_name, fourcc, fps_of_video, (w, h))
        # list_file = open('output/detection.txt', 'w')
        # frame_index = -1

    csv_file_path = os.path.join('outputs', input_video_name + '_' + str(fps_of_video) + '.csv')
    csv_file = open(csv_file_path, "w")
    csv_file.write('TrackerID,Class,Xmin,Ymin,Xmax,Ymax,FrameNumber\n')

    fps = 0.0
    frame_num = 0
    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        frame_num += 1
        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            # if track.track_id not in id_and_se_bbox_list:
            #     id_and_se_bbox_list[track.track_id] = [bbox, []]
            # else:
            #     id_and_se_bbox_list[track.track_id][1] = bbox

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)
            if len(class_names) > 0:
                class_name = class_names[0][0]
                cv2.putText(frame, class_name, (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)
                csv_file.write(
                    str(track.track_id) + ',' + class_name + ',' + str(int(bbox[0])) + ',' + str(int(bbox[1]))
                    + ',' + str(int(bbox[2])) + ',' + str(int(bbox[3])) + ',' + str(frame_num) + '\n')

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            # center point
            cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        cv2.putText(frame, "Total Vehicle Count: " + str(count), (20, 100), 0, 5e-3 * 200, (20, 255, 5), 3)
        # cv2.putText(frame, "Current Object Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (20, 40), 0, 5e-3 * 150, (20, 255, 5), 2)
        # cv2.namedWindow("AHEAD - YOLO3 Deep SORT", 0)
        # cv2.resizeWindow('AHEAD - YOLO3 Deep SORT', 1024, 768)
        # cv2.imshow('AHEAD - YOLO3 Deep SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            # frame_index = frame_index + 1
            # list_file.write(str(frame_index) + ' ')
            # if len(boxs) != 0:
            #     for i in range(0, len(boxs)):
            #         list_file.write(
            #             str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            # list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2
        # print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Finish", frame_num)
    end = time.time()
    csv_file.close()

    # if len(pts[track.track_id]) is not None:
    #     print(args["input"][43:57] + ": " + str(count) + " " + str(class_name) + ' Found')
    # else:
    #     print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        # list_file.close()
    cv2.destroyAllWindows()

    # for x in id_and_se_bbox_list:
    #     print(x, ': ', id_and_se_bbox_list[x])


if __name__ == '__main__':
    main(YOLO())
