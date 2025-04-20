import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from collections import deque
from collections import defaultdict
import random
from sklearn.cluster import KMeans
# Assign unique color per object ID
id_colors = {}
import numpy as np
def get_color_for_id(obj_id):
    if obj_id not in id_colors:
        id_colors[obj_id] = tuple(random.randint(0, 255) for _ in range(3))
    return id_colors[obj_id]
fps_deque = deque(maxlen=30)  # track last 30 frames
import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

def compute_slope_intercept(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return None, None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def compute_avg_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x_min = int(width * 0.1)        # 10% from left
    x_max = int(width * 0.9)        # 90% from left
    y_min = int(height * 0.4)       # 40% from top (i.e., 60% from bottom)
    y_max = height                  # Bottom of the image

    # Helper function to check if a point is within the ROI
    def in_roi(x, y):
        return x_min <= x <= x_max and y_min <= y <= y_max

    # Step 1: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Use Hough Line Transform to detect straight lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    # Step 4: Draw lines that match road-like angles
    selected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Filter lines with angles close to road edge direction
            if 20 < angle < 70 and in_roi(x1, y1) and in_roi(x2, y2):
                selected_lines.append(((x1, y1), (x2, y2)))

    # --- Extract slope and intercept for clustering ---


    features = []
    line_params = []

    for (x1, y1), (x2, y2) in selected_lines:
        slope, intercept = compute_slope_intercept(x1, y1, x2, y2)
        if slope is not None:
            features.append([slope, (x1 + x2) / 2])
            line_params.append((slope, intercept))

    # --- Apply KMeans to group lines into 2 clusters ---
    features = np.array(features)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    labels = kmeans.labels_

    # --- Average slope and intercept per cluster ---
    cluster_lines = {0: [], 1: []}
    for i, (slope, intercept) in enumerate(line_params):
        cluster_lines[labels[i]].append((slope, intercept))

    avg_lines = []
    for group in cluster_lines.values():
        if group:
            avg_slope = np.mean([s for s, _ in group])
            avg_intercept = np.mean([i for _, i in group])
            avg_lines.append((avg_slope, avg_intercept))

    # Sort lines based on their x-intercept at the bottom of the image (y = height)
    y_bottom = height  # Assuming `height` is the height of the image
    avg_lines.sort(key=lambda line: (y_bottom - line[1]) / line[0])  # Sort by x-intercept

    return avg_lines

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def velocity(id,coordinates, fps, avg_lines,image):
    # Extract slopes and intercepts of the two lines
    slope1, intercept1 = avg_lines[0]
    slope2, intercept2 = avg_lines[1]

    # Calculate the horizontal pixel distance between the two lines at the bottom of the image
    y_bottom = coordinates[-1]["center"][1]  # Use the y-coordinate of the last object's center
    x1 = (y_bottom - intercept1) / slope1
    x2 = (y_bottom - intercept2) / slope2


    pixel_distance = abs(x2 - x1)

    # Convert pixel distance to meters per pixel
    meters_per_pixel = 4 / pixel_distance  # 4 meters is the real-world distance

    # Calculate the distance traveled in pixels between frames
    distance_traveled_between_frames = []
    for i in range(1, len(coordinates)):
        x_now, y_now = coordinates[i]["center"]
        x_before, y_before = coordinates[i - 1]["center"]
        distance = ((x_now - x_before) ** 2 + (y_now - y_before) ** 2) ** 0.5
        distance_traveled_between_frames.append(distance)

    # Calculate the average velocity in pixels per second
    n = len(distance_traveled_between_frames)
    if n > 0:
        weighted_sum = sum(distance_traveled_between_frames[i] for i in range(n))/n
        avg_pixel_velocity = fps * weighted_sum
    else:
        avg_pixel_velocity = 0

    # Convert pixel velocity to meters per second
    avg_meter_velocity = avg_pixel_velocity * meters_per_pixel

    # Convert meters per second to kilometers per hour
    avg_velocity_kmph = avg_meter_velocity * 3.6

    return avg_velocity_kmph


up_count = 0
down_count = 0
car_count = 0
truck_count = 0
tracker1 = []
tracker2 = []
object_coords = defaultdict(list)

dir_data = {}

# Initialize global variables to store running averages
running_avg_slope = [[],[]]
running_avg_intercept = [[],[]]

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.weights, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        # print(f"Image: {img.shape} ")
        # print(f"Image Type: {type(img)} ")
        
        t1 = time_sync()


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1


        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            start_time = time.time()
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p) 
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            # print(f"W: {w} h {h}")F
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # Compute average lines
                height, width, _ = im0.shape

                avg_lines = compute_avg_lines(im0)
                avg_lines_prime = [[],[]]
                y1 = height
                y2 = int(height * 0.4)  # top of ROI (y_min)

                for line_grp  in range(0,len(avg_lines)):
                    slope, intercept = avg_lines[line_grp]
                    running_avg_slope[line_grp].append(slope)
                    running_avg_intercept[line_grp].append(intercept)
                    intercept_to_use = sum(running_avg_intercept[line_grp])/len(running_avg_intercept[line_grp])
                    slope_to_use = sum(running_avg_slope[line_grp])/len(running_avg_slope[line_grp])

                    x1 = int((y1 - intercept_to_use) / slope_to_use)
                    x2 = int((y2 - intercept_to_use) / slope_to_use)
                    cv2.line(im0, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue color for the lines
                    avg_lines_prime[line_grp] = [slope_to_use, intercept_to_use]

                im0 = cv2.resize(im0, (1000,700))
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        # Track coordinates for each object ID



                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # print(f"Img: {im0.shape}\n")
                        _dir =  direction(id,bboxes[1])

                        #count
                        count_obj(bboxes,w,h,id,_dir,int(cls))
                        # print(im0.shape)

                        # Draw object trail (center points)



                        bbox_center = (int((bboxes[0] + bboxes[2]) / 2), int((bboxes[1] + bboxes[3]) / 2))
                        object_coords[id].append({
                            "frame": frame_idx,
                            "center": bbox_center,
                            "bbox": [int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])]
                        })

                        pts = object_coords[id]
                        if len(pts) >= 2:
                            for i in range(1, len(pts)):
                                cv2.line(
                                    im0,
                                    pts[i - 1]["center"],
                                    pts[i]["center"],
                                    get_color_for_id(id),
                                    thickness=2
                                )

                        c = int(cls)  # integer class
                        if len(object_coords[id]) > 2:
                            velocity_of_id = velocity(id,object_coords[id], vid_cap.get(cv2.CAP_PROP_FPS),avg_lines_prime,im0)
                        else:
                            velocity_of_id = 0
                        label = f'{id} {names[c]} {velocity_of_id :.2f} km/h'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
                    

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')




            # Stream results
            im0 = annotator.result()
            if show_vid:
                global up_count,down_count
                color=(0,0,255)
                # print(f"Shape: {im0.shape}")

                # Left Lane Line
                # cv2.line(im0, (0, h-300), (600, h-300), (255,0,0), thickness=3)

                # Right Lane Line
                # cv2.line(im0,(680,h-300),(w,h-300),(0,0,255),thickness=3)
                
                thickness = 3 # font thickness
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1.2 
                # cv2.putText(im0, "Outgoing Traffic:  "+str(up_count), (60, 150), font, 
                #    fontScale, (0,0,255), thickness, cv2.LINE_AA)

                # cv2.putText(im0, "Incoming Traffic:  "+str(down_count), (700,150), font, 
                #    fontScale, (255,0,0), thickness, cv2.LINE_AA)
                
                # -- Uncomment the below lines to computer car and truck count --
                # It is the count of both incoming and outgoing vehicles 
                
                #Objects 
                # cv2.putText(im0, "Cars:  "+str(car_count), (60, 250), font, 
                #    1.5, (20,255,0), 3, cv2.LINE_AA)                

                # cv2.putText(im0, "Trcuks:  "+str(truck_count), (60, 350), font, 
                #    1.5, (20,255,0), 3, cv2.LINE_AA)  
                
                end_time = time.time()
                time_taken = end_time - start_time
                if time_taken == 0:

                    fps = 0
                else:
                    fps = 1/time_taken
                
                fps_deque.append(1 / (end_time - start_time))
                cv2.putText(im0, f"source FPS: {vid_cap.get(cv2.CAP_PROP_FPS)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(im0, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                try :
                    cv2.imshow('Walia Traffic Management', im0)
                    if cv2.waitKey(1) % 256 == 27:  # ESC code 
                        raise StopIteration  
                except KeyboardInterrupt:
                    raise StopIteration
                

            # Save results (image with detections)
            if save_vid or True:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1000,700))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)




def count_obj(box,w,h,id,direct,cls):
    global up_count,down_count,tracker1, tracker2, car_count, truck_count
    cx, cy = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))



    # For South

    if cy<= int(h//2):
        return

    if direct=="South":

        if cy > (h - 300):
            if id not in tracker1:
                # print(f"\nID: {id}, H: {h} South\n")
                down_count +=1
                tracker1.append(id)

                if cls==2:
                    car_count+=1
                elif cls==7:
                    truck_count+=1
            
    elif direct=="North":
        if cy < (h - 150):
            if id not in tracker2:
                # print(f"\nID: {id}, H: {h} North\n")
                up_count +=1
                tracker2.append(id)
                
                if cls==2:
                    car_count+=1
                elif cls==7:
                    truck_count+=1


def direction(id,y):
    global dir_data

    if id not in dir_data:
        dir_data[id] = y
    else:
        diff = dir_data[id] -y

        if diff<0:
            return "South"
        else:
            return "North"


if __name__ == '__main__':
    __author__ = 'Mogs'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default=r"C:\Users\amogh\OneDrive\Desktop\CAMSTUFF\shopping_multiple.H265", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default='store_true', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
