import numpy as np
import cv2
from collections import deque
from typing import List, Dict, Tuple
import time
import AP3
import math

video_time = 0

def euclidean_distance(v1,v2):
    return math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(v1, v2)))

class Object:
    def __init__(self, centroid, prev_cen_buf=deque(maxlen=10)):
        self.centroid = centroid
        self.prev_cen_buf = prev_cen_buf
        self.prev_cen_times = deque(maxlen=prev_cen_buf.maxlen)
        self.creation_time = video_time

    def get_velocity(self, pixels_per_meter):
        self.prev_cen_buf.append(self.centroid)
        self.prev_cen_times.append(video_time)

        if len(self.prev_cen_buf) == 1:
            return 0

        velocities = []
        for i in range(1, len(self.prev_cen_buf)):
            deltatime = self.prev_cen_times[i] - self.prev_cen_times[i - 1]
            distance = euclidean_distance(self.prev_cen_buf[i-1], self.prev_cen_buf[i])
            distance_km = distance / (pixels_per_meter * 1000)
            velocity = (distance_km/deltatime)*3600
            velocities.append(velocity)

        return np.average(velocities)

class Region:
    def __init__(self, points):
        self.points = points
        if not points:
            self.bbox = [0, 0, 0, 0]
            self.area = 0
            self.centroid = [0, 0]
            return
            
        # Calculate bounding box
        x_coords = [p[1] for p in points]
        y_coords = [p[0] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        self.bbox = [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]
        
        # Calculate centroid
        self.centroid = [int(sum(x_coords)/len(x_coords)), 
                        int(sum(y_coords)/len(y_coords))]
        
        # Calculate area of bbox
        self.area = math.sqrt(self.bbox[2] * self.bbox[3])

class Tracker:
    def __init__(self, pixels_per_meter, velocity_buffer=10, max_disappeared=10, max_distance=100, merge_threshold=5):
        self.next_id = 0
        self.objects = {}  # id -> Object
        self.disappeared = {}  # id -> frames_missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.pixels_per_meter = pixels_per_meter
        self.velocity_buffer = velocity_buffer
        self.merge_threshold = merge_threshold
        
    def update(self, regions: List[Region]) -> List[Dict]:
        if not regions:
            # Mark all as disappeared
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return []
            
        # Get current centroids
        centroids = [r.centroid for r in regions]
        
        # If no objects yet, register all
        if not self.objects:
            for centroid in centroids:
                self.objects[self.next_id] = Object(centroid, prev_cen_buf=deque(maxlen=self.velocity_buffer))
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            # Match objects to centroids
            obj_ids = list(self.objects.keys())
            old_objects = list(self.objects.values())

            # Calculate distances between all pairs
            D = []
            for old_objs in old_objects:
                row = []
                for cent in centroids:
                    dx = old_objs.centroid[0] - cent[0]
                    dy = old_objs.centroid[1] - cent[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    row.append(dist)
                D.append(row)
                
            # Match using minimum distances
            used_old = set()
            used_new = set()
            matches = []
            
            while True:
                min_dist = float('inf')
                min_i = min_j = -1
                
                for i in range(len(old_objects)):
                    if i in used_old:
                        continue
                    for j in range(len(centroids)):
                        if j in used_new:
                            continue
                        if D[i][j] < min_dist:
                            min_dist = D[i][j]
                            min_i = i
                            min_j = j
                            
                if min_dist > self.max_distance:
                    break
                    
                matches.append((min_i, min_j))
                used_old.add(min_i)
                used_new.add(min_j)
                
            # Update matched objects
            for old_idx, new_idx in matches:
                obj_id = obj_ids[old_idx]
                distance = np.linalg.norm(np.array(self.objects[obj_id].centroid) - np.array(centroids[new_idx]))
                self.objects[obj_id].centroid = centroids[new_idx]
                self.disappeared[obj_id] = 0
            
            # Mark unmatched objects as disappeared
            for i in range(len(old_objects)):
                if i not in used_old:
                    obj_id = obj_ids[i]
                    self.disappeared[obj_id] += 1
                    #old_objects[i].prev_cen_buf.clear()
                    if self.disappeared[obj_id] > self.max_disappeared:
                        del self.objects[obj_id]
                        del self.disappeared[obj_id]
            
            # Register new objects
            for j in range(len(centroids)):
                if j not in used_new:
                    self.objects[self.next_id] = Object(centroids[j], prev_cen_buf=deque(maxlen=self.velocity_buffer))
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
        
        # Prepare output
        results = []
        for obj_id, obj in self.objects.items():
            for region in regions:
                if region.centroid == obj.centroid:
                    results.append({
                        'id': obj_id,
                        'bbox': region.bbox,
                        'centroid': obj.centroid,
                        'area': region.area,
                        'velocity': obj.get_velocity(self.pixels_per_meter)
                    })
                    break
        return results

class MotionDetector:
    def __init__(self, min_area=500, threshold=25, noise_threshold=0.1, tracker=Tracker(50, 10, 10, 100), image_scale=1, image_smear=1, history_frames=1):
        self.min_area = min_area
        self.threshold = threshold
        self.tracker = tracker
        tracker.pixels_per_meter *= image_scale
        self.image_scale = image_scale
        self.image_smear = image_smear
        self.history_frames = history_frames
        self.prev_frames = []
        self.noise_threshold = noise_threshold

    def find_regions(self, image, connectivity=8):
        if image.size == 0:
            return []

        height, width = image.shape
        visited = np.zeros_like(image, dtype=bool)
        regions = []

        # Pre-compute neighbor offsets based on connectivity
        if connectivity == 8:
            neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        else:  # connectivity == 4
            neighbors = [(-1,0), (0,-1), (0,1), (1,0)]

        # Use a more efficient numpy-based approach
        # Find all non-zero points as starting candidates
        y_coords, x_coords = np.nonzero(image)

        for start_y, start_x in zip(y_coords, x_coords):
            if visited[start_y, start_x]:
                continue
                
            # Initialize region growing from this point
            region = []
            stack = [(start_y, start_x)]

            # Grow region using stack-based flood fill
            while stack:
                y, x = stack.pop()

                if visited[y, x]:
                    continue
                    
                visited[y, x] = True
                region.append([y, x])

                # Check all neighbors efficiently
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx

                    # Bounds check and validation in single condition
                    if (0 <= ny < height and 0 <= nx < width and 
                        image[ny, nx] > 0 and not visited[ny, nx]):
                        stack.append((ny, nx))

            # Only add regions that meet the minimum area requirement
            reg = Region(region)
            if reg.area >= self.min_area * (self.image_scale ** 2):
                regions.append(reg)

        return regions

    def gaussian_kernel(self, size, sigma=1.0):
        # Ensure size is odd
        size = int(size)
        if size % 2 == 0:
            size += 1

        # Create coordinate arrays
        x = np.linspace(-(size//2), size//2, size)
        y = np.linspace(-(size//2), size//2, size)
        x, y = np.meshgrid(x, y)

        # Calculate Gaussian values
        gaussian = np.exp(-(x**2 + y**2)/(2*sigma**2))

        # Normalize the kernel
        return gaussian / gaussian.sum()

    def preprocess_frame(self, frame, i=0):
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        prevs = self.prev_frames[i]

        # Simple motion detection using frame differencing
        if len(prevs) == 0:
            for j in range(self.history_frames):
                prevs.append(gray)
            
        # Calculate absolute difference
        diffs = [cv2.absdiff(gray, prev) for prev in prevs]
        diff = diffs[0]

        for j in range(1, len(diffs) - 1):
            diff = cv2.bitwise_or(diffs[j], diffs[j+1])

        # 0 1 2
        #-1 0 1

        #2 3 0 1
        for j in range(len(prevs)-1, 0, -1):
            prevs[j] = prevs[j-1]
        
        prevs[0] = gray
        
        sharp_kernel = np.array([[-1, -1, -1],
                                 [-1,  9, -1],
                                 [-1, -1, -1]]) * -1

        smear_kernel = np.array([[0, 0, 1, 0, 0],
                                 [0, 2, 4, 2, 0],
                                 [1, 4, 8, 4, 1],
                                 [0, 2, 4, 2, 0],
                                 [0, 0, 1, 0, 0]]) * self.image_smear

        leak_kernel = np.array([[8, 8, 8, 8, 8],
                                  [8, 4, 2, 4, 8],
                                  [8, 2, 1, 2, 8],
                                  [8, 4, 2, 4, 8],
                                  [8, 8, 8, 8, 8]])
        
        edges = cv2.absdiff(cv2.filter2D(gray, -1, sharp_kernel*(-1)), gray)
        
        proccessed = diff
        mask = proccessed >= 255*self.noise_threshold
        proccessed[~mask] = 0
        
        #proccessed = cv2.filter2D(proccessed, -1, smear_kernel)
        #proccessed = cv2.filter2D(proccessed, -1, self.gaussian_kernel(3, 3))
        proccessed = cv2.bitwise_and(proccessed, edges)
        proccessed = cv2.filter2D(proccessed, -1, self.gaussian_kernel(3, 3))
        proccessed = cv2.filter2D(proccessed, -1, smear_kernel)
        proccessed = cv2.filter2D(proccessed, -1, leak_kernel)
        
        return proccessed

    def process_frame(self, frame):
        b, g, r = cv2.split(frame)
        
        # For three channels
        for i in range(3):
            self.prev_frames.append(deque(maxlen=self.history_frames))

        r = self.preprocess_frame(r, 0)
        g = self.preprocess_frame(g, 1)
        b = self.preprocess_frame(b, 2)

        result = []

        if len(r) != 0:
            result = cv2.bitwise_or(cv2.bitwise_or(np.array(b), np.array(g)), np.array(r))
        
        proccessed = np.array(result)
        if len(proccessed) != 0:
            cv2.imshow("proccessed", proccessed)
            
        # Threshold
        _, thresh = cv2.threshold(proccessed, self.threshold, 255, cv2.THRESH_BINARY)

        regions = []
        if thresh is not None:
            regions = self.find_regions(thresh)
        
        return self.tracker.update(regions)
    
    def process_video(self, video_path, output_path=None, target_fps=30):
        cap = cv2.VideoCapture(video_path)
        old_size = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
        new_size = np.multiply(old_size, self.image_scale)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frametime = 1/target_fps

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 30, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            resized_frame = AP3.nearest_neighbor_interpolation(frame, new_size[0], new_size[1])
            if not ret:
                break

            # Process frame
            moving_objects = self.process_frame(resized_frame)

            frame_end = time.time()
            ellapsed = frame_end - frame_start
            
            global video_time
            video_time += frametime
            if(ellapsed < 1/target_fps):
                time.sleep(1/target_fps - ellapsed)

            # Draw results
            for obj in moving_objects:
                bbox = np.array(obj['bbox']) / self.image_scale
                [x, y, w, h] = np.frompyfunc(int, 1, 1)(bbox)
                centroid = np.array(obj['centroid']) / self.image_scale
                centroid = [int(centroid[0]), int(centroid[1])]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {obj['id']} bbox: {w}x{h}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Velocity: {(obj['velocity']):.2f} km/h", (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if output_path:
                out.write(frame)

            cv2.imshow('Motion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MotionDetector(min_area=80, threshold=25, noise_threshold=0.3, tracker=Tracker(40, 5, 25, 100), image_scale=0.5)
    video_path = "physics.mp4"
    #detector.process_video(video_path)#, f"output_{video_path}.avi")
    
    detector = MotionDetector(min_area=150, threshold=20, noise_threshold=0.8, tracker=Tracker(40, 5, 25, 100), image_scale=0.3)
    video_path = "dp.mp4"
    #detector.process_video(video_path)#, f"output_{video_path}.avi")
    
    detector = MotionDetector(min_area=150, threshold=10, noise_threshold=0.25, tracker=Tracker(40, 5, 25, 100), image_scale=0.5, history_frames=4)
    video_path = "traffic2.mp4"
    #detector.process_video(video_path)#, f"output_{video_path}.avi")