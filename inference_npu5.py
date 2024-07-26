import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
from imutils.video import FPS
import time
import lib.config as config
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#cam = use gstreamer, cam2 = use cv2.videocapture
ap.add_argument("-i", "--inputtype", required=False, default="cam2",
    help="Select input cam, cam2, file")
ap.add_argument("-f", "--filename", required=False, default="skyfall.mp4",
    help="file video (.mp4)")
args = vars(ap.parse_args())

IMG_SIZE = config.IMG_SIZE

CLASSES = config.CLASSES

# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = config.DEVICE_COMPATIBLE_NODE

RK356X_RKNN_MODEL = config.RK356X_RKNN_MODEL
RK3588_RKNN_MODEL = config.RK3588_RKNN_MODEL

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK356x'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(input[..., 2:4]*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for i, pred in enumerate(input_data):
        pred = pred.squeeze()
        box_xy = pred[..., 0:2]
        box_wh = pred[..., 2:4]
        score = pred[..., 4:5]
        cls = pred[..., 5:]
        
        box_xy = box_xy * 2.0 - 0.5
        box_wh = (box_wh * 2) ** 2
        box_xy -= box_wh / 2
        
        box = np.concatenate((box_xy, box_wh), axis=-1)
        
        box, cls, score = filter_boxes(box, cls, score)
        boxes.append(box)
        classes.append(cls)
        scores.append(score)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):


    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    
    if args["inputtype"] == 'cam':
        gst_str = ("uvch264src device=/dev/video{} ! "
               "image/jpeg, width={}, height={}, framerate=30/1 ! "
               "jpegdec ! "
               "video/x-raw, format=BGR ! "
               "appsink").format(dev, width, height)
        vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    elif args["inputtype"] == 'file':
        gst_str = ("filesrc location={} ! "
               "qtdemux name=demux demux. ! queue ! faad ! audioconvert ! audioresample ! autoaudiosink demux. ! "
               "avdec_h264 ! videoscale ! videoconvert ! "
               "appsink").format(args["filename"])        
        vs = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    elif args["inputtype"] == 'cam2':
        vs = cv2.VideoCapture(dev)
        vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    return vs

if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK356x':
        rknn_model = RK356X_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Create Stream from Webcam
    vs = open_cam_usb(config.CAM_DEV, config.CAM_WIDTH, config.CAM_HEIGHT)

    time.sleep(2.0)
    fps = FPS().start()

    if not vs.isOpened():
        print("Cannot capture from camera. Exiting.")
        quit()

    prev_frame_time = 0
    new_frame_time = 0

    # loop over the frames from the video stream
    while True:
        ret, frame = vs.read()

        if not ret:
            break

        new_frame_time = time.time()
        show_fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        show_fps = int(show_fps)
        show_fps = str("{} FPS".format(show_fps))

        ori_frame = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, ratio, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))

        # Convert to 4D array (N, C, H, W)
        frame = np.expand_dims(frame, axis=0)

        # Inference
        outputs = rknn_lite.inference(inputs=[frame])

        # # 첫 번째 프레임에서만 출력 형태 확인
        # if fps._numFrames == 0:
        #     outputs = rknn_lite.inference(inputs=[frame])
        #     print("Number of outputs:", len(outputs))
        #     for i, output in enumerate(outputs):
        #         print(f"Output {i} shape:", output.shape)

        # 단일 출력 처리
        input_data = outputs[0]  # 단일 출력 사용

        # reshape 불필요, 직접 사용
        input_data = [input_data]  # list 형태로 유지

        # post process
        boxes, classes, scores = yolov5_post_process(input_data)
        img_1 = ori_frame

        if boxes is not None:
            draw(img_1, boxes, scores, classes)
           
            # show FPS in Frame
            cv2.putText(img_1, show_fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
            
            # show output
            cv2.imshow("yolov5 post process result", img_1)

            
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        

        # update the FPS counter
        fps.update()

    rknn_lite.release()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()
