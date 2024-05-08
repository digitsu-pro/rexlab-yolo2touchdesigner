import argparse, queue, threading, json, time, os.path
import cv2, torch

from termcolor import colored
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO

print(colored('Start service', 'green'))

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8l",
        help="Model name which should be used for task: yolov8n, yolov8m, yolov8l, yolov8x")
    parser.add_argument("--ip", default="127.0.0.1",
        help="The ip of the OSC server from TouchDesigner")
    parser.add_argument("--port", type=int, default=5005,
        help="The port of the OSC server is listening on from TouchDesigner")
    parser.add_argument("--stream", default=r"D:\ultralytics\video-example.mp4",
        help="Video stream URL for inference (RTSP)")

    parser.add_argument("--debug", action='store_true',
        help="Print debug output for each frame")
    parser.add_argument("--show", action='store_true',
        help="Show separate window with frames tracking")
    
    parser.add_argument("--confidence", type=float, default=0.1,
        help="Minimum model confidence for object tracking (range from 0.0 to 1.0). Default: 0.1")
    parser.add_argument("--tracking_period", type=int, default=1,
        help="Number of frames between new trackings. Default: 1 (each frame with tracking)")
    parser.add_argument("--objects_max", type=int, default=10,
        help="Maximum objects detections. Default: 10")
    parser.add_argument("--objects_filter", default="",
        help="Filter objects by class name. Work when --single_class is not defined.")
    parser.add_argument("--object_persistance", type=int, default=10,
        help="Filter objects base on time (in ms). Default: 10 ms")
    parser.add_argument("--single_class", type=int, default=-1,
        help="Objects class to detect (ignore all other objects). Default: 0 - person")

    args = parser.parse_args()

print(colored("Info: check for CUDA availability...", "blue"))
if torch.cuda.is_available():
    print(colored(" - CUDA available, inference will run fast on GPU", "green"))
else:
    print(colored(" - CUDA not found, inference will run slower on CPU", "yellow"))


class ObjectsBuffer:
    def __init__(self, size=10):
        self.objects = {}
        self.maxSize = size

        self.setup()

    def setup(self):
        for i in range(self.maxSize):
            self.objects[f"p{i+1}"] = { 'track_id': -1, 'time': 0, 'free': True, 'index': i+1, 'center': [0, 0] }

    def free(self):
        for id in self.objects:
            self.objects[id]['free'] = True
    
    def found(self, track_id):
        exists = False

        for id in self.objects:
            if self.objects[id]['track_id'] == track_id:
                self.objects[id]['free'] = False

                if self.objects[id]['time'] == 0:
                    self.objects[id]['time'] = time.time()

                exists = True

                break

        return exists

    def add(self, track_id):
        index = 0

        if self.found(track_id):
            return index

        for id, obj in self.objects.items():
            if obj['free']:
                obj['track_id'] = track_id
                obj['free'] = False
                obj['time'] = time.time()
                index = obj['index']
                break

        return index

    def set_center(self, track_id, box):
        if not 'x1' in box or not 'x2' in box or not 'y1' in box or not 'y2' in box:
            return

        centerX = (box['x1'] + box['x2']) / 2
        centerY = (box['y1'] + box['y2']) / 2

        for id, obj in self.objects.items():
            if obj['track_id'] == track_id:
                obj['center'] = [centerX, centerY]
                break

    def each(self):
        for id in self.objects:
            yield self.objects[id]
    
    def reset_time(self):
        for id in self.objects:
            if self.objects[id]['free']:
                self.objects[id]['time'] = 0

    def dump(self):
        print(self.objects)



# OSC worker class
class OSCWorker:
    # - establish connection with TouchDesigner OSC server
    def __init__(self, args):
        self.ip = args.ip
        self.port = args.port
        self.debug = args.debug

        self.confidence = args.confidence
        self.objectsFilter = args.objects_filter
        self.objectPersistance = args.object_persistance
        self.objectsBuf = ObjectsBuffer(args.objects_max)

        while True:
            try:
                self.client = SimpleUDPClient(self.ip, self.port)
                self.status = True
                print(colored("Info: UDP client ready to send data", "green"))
                break
            except Exception as e:
                self.status = False
                print(colored(f"Warning: connection failed ({e}), retrying in {args.timeout} seconds...", "yellow"))
                time.sleep(args.timeout)

    def send_tracking_data(self, detections):
        now = time.time()

        # Filter and sort detections
        if args.single_class < 0:
            detections = [item for item in detections if item["name"] in self.objectsFilter]
        
        detections = sorted(detections, key=lambda x: (-x["confidence"], x.get("track_id", float('inf'))))

        # Free all objects
        self.objectsBuf.free()

        # First, find all prev track_ids
        for detected in detections:
            if "track_id" in detected:
                self.objectsBuf.found(detected["track_id"])
        
        # Second, add new track_ids
        for detected in detections:
            if detected['confidence'] < self.confidence or not "track_id" in detected:
                continue

            self.objectsBuf.add(detected['track_id'])

            self.objectsBuf.set_center(detected['track_id'], detected['box'])

        if self.debug:
            self.objectsBuf.dump()

        self.objectsBuf.reset_time()

        # Third, send data from buffer to OSC server
        for object in self.objectsBuf.each():
            objectPersist = (now - object['time']) * 1000
            if not object['free'] and object['track_id'] > 0 and objectPersist >= self.objectPersistance:
                self.send(f"/p{object['index']}_x", object['center'][0])
                self.send(f"/p{object['index']}_y", object['center'][1])
    
    # - method to send data
    def send(self, chanel, data):
        if self.client is not None:
            self.client.send_message(chanel, data)

class CaptureThread:
    def __init__(self, src):
        print(colored("Info: create new daemon thread for stream capturing", "blue"))

        self.lock = threading.Lock()
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.ready = False
        self.waiting = 0

        self.q = queue.Queue()
        self.thread_run = threading.Thread(target=self.run)
        self.thread_run.daemon = True

        if "://" in f"{src}":
            self.type = 'stream'
        else:
            self.type = 'video'

        self.thread_run.start()

        self.init()

        print(colored("Info: capture tread started", "blue"))

    def init(self):
        while True:
            if self.ready:
                break

            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.src)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            time.sleep(3)

    def run(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                if self.waiting == 0:
                    self.ready = False
                    self.waiting = time.time()
            
                waited = int(time.time() - self.waiting)

                if waited > 0:
                    print(colored(f"Pause: waiting for stream start {waited} s", "yellow"), end="\r", flush=True)

                continue
            
            if self.waiting > 0:
                self.waiting = 0

            start_time = time.time()
            # Grab the next frame
            try:
                grabbed = self.cap.grab()
            except:
                continue

            if not grabbed:
                continue

            self.ready = True

            # Retrieve and decode the frame
            ret, frame = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            
            # Wait until the next frame should be displayed for video file
            if self.type == 'video':
                time.sleep(max(1./fps - (time.time() - start_time), 0))

    def detect_fps(self):
        # Find OpenCV version and get FPS
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            self.fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(colored(f"Info: detected source FPS - {fps}", "blue"))

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

class Tracker:
    def __init__(self, args):
        # Load a model
        self.model_task = "detection"
        self.model_name = args.model
        self.single_class = args.single_class
        self.debug = args.debug

        self.model_path = f"{self.model_name}.pt"

        print(colored(f"Info: loading YOLO model {self.model_name}", "blue"))

        if os.path.isfile(f"../models/{self.model_name}.pt"):
            self.model_path = f"../models/{self.model_name}.pt"

        self.model =YOLO(model=self.model_path, task=self.model_task, verbose=self.debug)
    
    def warm_up(self, cap):
        print(colored("Info: warm up the model on first frames", "blue"))

        frames = 10
        ret = None

        while not ret or frames > 0:
            ret, frame = cap.read()

            if not ret:
                continue
            
            self.process_frame(frame)
            frames -= 1
        
        print(colored("Info: warm up succesfull, ready", "blue"))

    def process_frame(self, frame):
        # Run inference on frame
        if self.single_class >= 0:
            results = self.model.track(source=frame, persist=True, show=False, verbose=self.debug, classes=self.single_class)
        else:
            result = self.model.track(source=frame, persist=True, show=False, verbose=self.debug)

        # Debugging: Print track IDs
        if args.debug:
            print('- read frame and run inference')
            if results[0].boxes.id is not None:
                print("Track IDs:", results[0].boxes.id)

        return results


# Initialize OSC UDP client
osc_worker = OSCWorker(args)

# Initialize model wrapper for tracking
tracker = Tracker(args)

# Initialize and start the stream capture thread
stream = CaptureThread(args.stream)

print(colored("Info: all things seems to be ready, starting main loop...", "blue"))
try:
    while not stream.ready:
        continue
    
    tracker.warm_up(stream)
    frame_index = 0

    while True:
        start_time = time.time()
        ret, frame = stream.read()

        if not ret:
            if stream.type == 'stream':
                continue
            else:
                print(colored('Warning: end of frames', 'yellow'))
                stream.stop()
                stream.join()
                cv2.destroyAllWindows()
                break

        frame_index += 1
        if frame_index >= args.tracking_period:
            frame_index = 0
        else:
            continue

        # Process the frame
        results = tracker.process_frame(frame)
        detections_json = results[0].tojson()
        detections = json.loads(detections_json)

        osc_worker.send_tracking_data(detections)

        if args.show:
            annotated_frame = results[0].plot()
            #annotated_frame = cv2.resize(annotated_frame, (show_width, show_height))
            cv2.imshow("Tracking results", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    print(colored('Info: close video stream capture', 'blue'))
    stream.stop()
    cv2.destroyAllWindows()







