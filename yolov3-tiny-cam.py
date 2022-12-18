import time
import cv2
import onnxruntime
import numpy as np
import threading
import queue
import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itchat


class Yolov3TinyCam:
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    session = onnxruntime.InferenceSession("tiny-yolov3.onnx")
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    label = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
             "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
             "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
             "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
             "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
             "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
             "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    queue = queue.Queue(maxsize=100)
    async_job = None
    exit = False

    def __init(self):
        signal.signal(signal.SIGINT, self.leave_now)
        signal.signal(signal.SIGTERM, self.leave_now)

    def frame_process(self, frame, input_shape=(416, 416)):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_shape)
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def get_prediction(self, image_data, image_size):
        input = {
            self.inname[0]: image_data,
            self.inname[1]: image_size
        }
        t0 = time.time()
        boxes, scores, indices = self.session.run(self.outname, input)
        predict_time = time.time() - t0
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices[0]:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])
        return out_boxes, out_scores, out_classes, predict_time

    def draw_box_on_frame(self, frame, out_boxes, out_scores, out_classes):
        plt.figure()
        px = 1 / plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1, figsize=(self.width*px, self.height*px))
        ax.imshow(frame)

        resized_width = self.width/412
        resized_height = self.height/412

        detected = False
        for i, l in enumerate(out_classes):
            if self.label[l] == 'person':
                box = out_boxes[i]
                y1, x1, y2, x2 = box[0] * resized_height, box[1] * resized_width, box[2] * resized_height, box[3] * resized_width
                color = 'blue'
                box_h = (y2 - y1)
                box_w = (x2 - x1)
                bbox = patches.Rectangle((y1, x1), box_h, box_w, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(y1, x1, s='person', color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})
                detected = True
        plt.axis('off')

        if detected:
            plt.savefig("output/%d.jpg" % int(time.time()), bbox_inches='tight', pad_inches=0.0)

    def draw_box_on_frame2(self, frame, out_boxes, out_scores, out_classes):
        resized_width = self.width/412
        resized_height = self.height/412

        detected = False
        for i, l in enumerate(out_classes):
            if self.label[l] == 'person':
                box = out_boxes[i]
                y1, x1, y2, x2 = box[0] * resized_height, box[1] * resized_width, box[2] * resized_height, box[3] * resized_width
                color = 'blue'
                box_h = (y2 - y1)
                box_w = (x2 - x1)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                detected = True
        if detected:
            cv2.imwrite("output/%d.jpg" % int(time.time()), frame)

    def run_async_job(self):
        def async_job():
            while True:
                try:
                    frame, out_boxes, out_scores, out_classes = self.queue.get()
                except TypeError:
                    break
                # example of inference result
                # [[191.84231567382812, 169.9823760986328, 230.2967071533203, 225.7290802001953]][0.6310255527496338][62]
                # box: [191.84231567382812, 169.9823760986328, 230.2967071533203, 225.7290802001953]
                # score: 0.6310255527496338, tvmonitor
                for i, c in reversed(list(enumerate(out_classes))):
                    print("box:", out_boxes[i])
                    print("score:", out_scores[i], ",", self.label[c])
                self.draw_box_on_frame2(frame, out_boxes, out_scores, out_classes)
        self.async_job = threading.Thread(target=async_job)
        self.async_job.start()

    def leave_now(self, signum, frame):
        self.exit = True

    def run_loop_forever(self):
        while self.cap.isOpened():
            # check async job thread state
            if self.async_job and self.async_job.is_alive():
                pass
            else:
                self.run_async_job()

            # read cam and inference with yolov3-tiny
            ret, frame = self.cap.read()
            if ret and not self.exit:
                image_data = self.frame_process(frame, input_shape=(416, 416))
                image_size = np.array([416, 416], dtype=np.float32).reshape(1, 2)
                out_boxes, out_scores, out_classes, predict_time = self.get_prediction(image_data, image_size)
                out_boxes = np.array(out_boxes).tolist()
                out_scores = np.array(out_scores).tolist()
                out_classes = np.array(out_classes).tolist()
                if not self.queue.full():
                    self.queue.put(tuple((frame, out_boxes, out_scores, out_classes)), block=False)
            else:
                self.queue.put(tuple())
                break
            time.sleep(1)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Yolov3TinyCam().run_loop_forever()