import time     # For keeping track of time
import os       # For storing results in files

import math
import numpy as np
import cv2

from collections import deque           #Data point storage#
from typing import List, Dict, Tuple    #######=SAME=########


###=GLOBAL=VARIABLES=###############

video_time = 0

log = ""

#####################################

###=GLOBAL=METHODS=###################

def euclidean_distance(v1,v2):
    return math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(v1, v2)))

def magnitude(xy):
    return math.sqrt(xy[0]**2 + xy[1]**2) 

def calculate_drag(mass, velocity):
    return np.array((world.C_d/mass) * magnitude(velocity) * velocity, dtype=np.float128)

def simulate_ball_motion_euler(pos, vel, mass, dt_init, t_max, tol=1e-3):
    x, y = pos
    vx, vy = vel
    
    # Initialize lists to store trajectories
    positions = [[x, y]]
    velocities = [[vx, vy]]
    t = 0.0
    dt = dt_init
    while t < t_max:
        # Take two steps with dt
        dt = min(dt, t_max - t)
        
        # Single step
        x1, y1 = x + vx * dt, y + vy * dt
        f_d1 = calculate_drag(mass, np.array([vx, vy]))
        vx1 = vx - f_d1[0] * dt
        vy1 = vy + (world.g + f_d1[1]) * dt
        
        # Two half steps
        dt_half = dt / 2
        xh, yh = x + vx * dt_half, y + vy * dt_half
        f_dh = calculate_drag(mass, np.array([vx, vy]))
        vxh = vx - f_dh[0] * dt_half
        vyh = vy + (world.g + f_dh[1]) * dt_half
        
        x2, y2 = xh + vxh * dt_half, yh + vyh * dt_half
        f_d2 = calculate_drag(mass, np.array([vxh, vyh]))
        vx2 = vxh - f_d2[0] * dt_half
        vy2 = vyh + (world.g + f_d2[1]) * dt_half
        
        # Calculate error estimates
        error_pos = max(abs(x1 - x2), abs(y1 - y2))
        error_vel = max(abs(vx1 - vx2), abs(vy1 - vy2))
        error = max(error_pos, error_vel)
        #print(dt)
        # Adjust step size based on error
        if error > tol:
            dt *= 0.5  # Reduce step size
            continue

        # Accept the step if error is within tolerance
        t += dt
        x, y = x2, y2
        vx, vy = vx2, vy2
        
        positions.append([x, y])
        velocities.append([vx, vy])
        
        # Increase step size if error is very small
        if error < tol/10:
            dt = min(dt * 2, dt_init)
    
    return np.array(positions), np.array(velocities), error_pos, error_vel

def simulate_ball_motion_rk4(pos, vel, mass, dt_init, t_max, tol=1e-3):
    x, y = pos
    vx, vy = vel
    
    positions = [[x, y]]
    velocities = [[vx, vy]]
    t = 0.0
    
    def derivatives(pos, vel):
        x, y = pos
        vx, vy = vel
        f_d = calculate_drag(mass, np.array(vel))
        ax = -f_d[0]
        ay = world.g + f_d[1]
        return np.array([vx, vy]), np.array([ax, ay])
    
    def rk4_step(pos, vel, dt):
        x, y = pos
        vx, vy = vel
        
        # k1
        k1_vel, k1_acc = derivatives([x, y], [vx, vy])
        
        # k2
        k2_pos = [x + 0.5 * k1_vel[0] * dt, y + 0.5 * k1_vel[1] * dt]
        k2_vel = [vx + 0.5 * k1_acc[0] * dt, vy + 0.5 * k1_acc[1] * dt]
        k2_vel, k2_acc = derivatives(k2_pos, k2_vel)
        
        # k3
        k3_pos = [x + 0.5 * k2_vel[0] * dt, y + 0.5 * k2_vel[1] * dt]
        k3_vel = [vx + 0.5 * k2_acc[0] * dt, vy + 0.5 * k2_acc[1] * dt]
        k3_vel, k3_acc = derivatives(k3_pos, k3_vel)
        
        # k4
        k4_pos = [x + k3_vel[0] * dt, y + k3_vel[1] * dt]
        k4_vel = [vx + k3_acc[0] * dt, vy + k3_acc[1] * dt]
        k4_vel, k4_acc = derivatives(k4_pos, k4_vel)
        
        # Update position and velocity
        dx = (dt / 6) * (k1_vel[0] + 2 * k2_vel[0] + 2 * k3_vel[0] + k4_vel[0])
        dy = (dt / 6) * (k1_vel[1] + 2 * k2_vel[1] + 2 * k3_vel[1] + k4_vel[1])
        dvx = (dt / 6) * (k1_acc[0] + 2 * k2_acc[0] + 2 * k3_acc[0] + k4_acc[0])
        dvy = (dt / 6) * (k1_acc[1] + 2 * k2_acc[1] + 2 * k3_acc[1] + k4_acc[1])
        
        return [x + dx, y + dy], [vx + dvx, vy + dvy]
    
    dt = dt_init
    while t < t_max:
        dt = min(dt, t_max - t)
        
        # Take one full step
        pos1, vel1 = rk4_step([x, y], [vx, vy], dt)
        
        # Take two half steps
        pos_half, vel_half = rk4_step([x, y], [vx, vy], dt/2)
        pos2, vel2 = rk4_step(pos_half, vel_half, dt/2)
        
        # Calculate error estimates
        error_pos = max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
        error_vel = max(abs(vel1[0] - vel2[0]), abs(vel1[1] - vel2[1]))
        error = max(error_pos, error_vel)
        
        # Adjust step size based on error
        if error > tol:
            dt *= 0.5  # Reduce step size
            continue
        
        # Accept the step if error is within tolerance
        t += dt
        x, y = pos2
        vx, vy = vel2
        
        positions.append([x, y])
        velocities.append([vx, vy])
        
        # Increase step size if error is very small
        if error < tol/10:
            dt = min(dt * 2, dt_init)
    
    return np.array(positions), np.array(velocities), error_pos, error_vel
    
def shooting_method(shot_ball, target_ball, dt, h_init=1.0, tol=1e-3, max_iter=-1, s=[0, 0]):
    f1_0 = shot_ball.center
    f1_1 = target_ball.center

    global log
    init_log = f"Initializing shooting method: {f1_0}, {f1_1}"
    log += init_log+'\n'  
    f2_0 = np.array(s, dtype=float)
    
    h = h_init
    prev_error = float('inf')

    cur_iter = 0
    
    done = False
    while not done:
        poses, vels, error_pos, error_vel = world.estimation_method(f1_0, f2_0, shot_ball.mass, dt, 1)
        f1_iter = poses[-1]
        f2_iter = vels[-1]
        
        # Calculate current error
        current_error = np.sqrt((f1_iter[0] - f1_1[0])**2 + (f1_iter[1] - f1_1[1])**2)
        
        # Check convergence
        if (current_error <= tol and max_iter == -1) or (current_error <= tol and max_iter != -1 and cur_iter >= max_iter):
            done = True
            break
            
        # Adapt step size based on error change
        error_ratio = current_error / prev_error if prev_error != float('inf') else 1.0
        prev_error = current_error
        
        if error_ratio < 0.95:
            h *= 0.5 / error_ratio
        elif error_ratio > 0.5:
            h *= 1.5 / error_ratio
            
        print(f"h:{h}, current: {current_error}")
        
        f2_0[0] += h if f1_iter[0] < f1_1[0] else -h
        f2_0[1] += h if f1_iter[1] < f1_1[1] else -h

        cur_iter+=1

    end_log = f"    real target position: {f1_1},  estimated target position:{f1_iter}\n"
    end_log +=f"    guessed initial velocity: {f2_0},  end velocity when shot ball meets target:{f2_iter}\n"
    end_log +=f"    Error:\n        estimation method:\n            position error: {error_pos},\n            velocity error: {error_vel},\n        shooting method: {current_error}\n\n"
    
    print(end_log)
    log += end_log
    
    return poses, vels

#####################################

###=CLASSES=#########################

class WorldProperties:
    def __init__(self):
        self.offset_threshold_x = 5
        self.offset_threshold_y = 5

        self.acceleration_step = 5

        self.C_d = 0.47/100    # drag coefficient of the sphere
        self.g = 0          # gravity (estimated later)

        self.estimation_method = simulate_ball_motion_rk4
        self.g_values = deque(maxlen=50)

    def update_gravity(self, g):
        if(np.isnan(g) or np.isinf(g) or g == 0):
            return

        self.g_values.append(g)
        self.g = self.get_avg_gravity()

    def get_avg_gravity(self):
        return np.average(self.g_values) if len(self.g_values) > 0 else 0

world = WorldProperties()

class Ball:
    def __init__(self, center, area, mass, color):
        self.center = np.array(center)
        self.radius = np.sqrt(area / np.pi)
        self.area = area
        self.color = color
        self.mass = mass
        
        self.velocity = np.array([0, 0])

    def calculate_drag(self):
        return calculate_drag(self.mass, self.velocity)

    def detect_collision(self, other):
        return euclidean_distance(self.center, other.center) < self.radius + other.radius

class Object:
    def __init__(self, centroid, radius, prev_cen_buf_x=deque(maxlen=1000), prev_cen_buf_y=deque(maxlen=1000)):
        self.centroid = centroid
        self.radius = radius
        
        self.prev_cen_buf_x = prev_cen_buf_x
        self.prev_cen_buf_y = prev_cen_buf_y
        
        self.prev_cen_x_times = deque(maxlen=prev_cen_buf_x.maxlen)
        self.prev_cen_y_times = deque(maxlen=prev_cen_buf_y.maxlen)
                
        self.velocities_x = deque(maxlen=prev_cen_buf_x.maxlen - 1)
        self.velocities_y = deque(maxlen=prev_cen_buf_y.maxlen - 1)

        self.velocities_times_x = deque(maxlen=prev_cen_buf_x.maxlen - 1)
        self.velocities_times_y = deque(maxlen=prev_cen_buf_y.maxlen - 1)

    def calculate_velocity(self):
        x, y = [0, 0]

        if len(self.prev_cen_buf_x) > 1:
            x = np.subtract(self.prev_cen_buf_x[-1], self.prev_cen_buf_x[-2]) / np.subtract(self.prev_cen_x_times[-1], self.prev_cen_x_times[-2])

        if len(self.prev_cen_buf_y) > 1:
            y = np.subtract(self.prev_cen_buf_y[-1], self.prev_cen_buf_y[-2]) / np.subtract(self.prev_cen_y_times[-1], self.prev_cen_y_times[-2])
        
        return [x, y]

    def calculate_acceleration(self):
        x, y = [0, 0]

        if len(self.velocities_x) > world.acceleration_step:
            x = np.subtract(self.velocities_x[-1], self.velocities_x[-world.acceleration_step-1]) / np.subtract(self.velocities_times_x[-1], self.velocities_times_x[-world.acceleration_step-1])

        if len(self.velocities_y) > world.acceleration_step:
            y = np.subtract(self.velocities_y[-1], self.velocities_y[-world.acceleration_step-1]) / np.subtract(self.velocities_times_y[-1], self.velocities_times_y[-world.acceleration_step-1])
        
        return [x, y]

    def get_last_velocity(self):
        return [self.velocities_x[-1] if len(self.velocities_x) > 0 else 0, self.velocities_y[-1] if len(self.velocities_y) > 0 else 0]

    def calculate_avg_velocity(self):
        return [np.average(self.velocities_x) if len(self.velocities_x) > 0 else 0, np.average(self.velocities_y) if len(self.velocities_y) > 0 else 0]

    def update_velocity_x(self):
        if len(self.prev_cen_buf_x) > 0 and abs(self.centroid[0] - self.prev_cen_buf_x[-1]) < world.offset_threshold_x:
            return

        self.prev_cen_buf_x.append(self.centroid[0])
        self.prev_cen_x_times.append(video_time)

        vel = np.round(self.calculate_velocity()[0], 2)

        if len(self.velocities_x) > 0 and self.velocities_x[-1] == vel:
            return

        #print(f"Velocity_x: {vel}")
        self.velocities_x.append(np.round(vel, 2))
        self.velocities_times_x.append(video_time)

    def update_velocity_y(self):
        if len(self.prev_cen_buf_y) > 0 and abs(self.centroid[1] - self.prev_cen_buf_y[-1]) < world.offset_threshold_y:
            return

        self.prev_cen_buf_y.append(self.centroid[1])
        #print(f"Pos_y: {self.centroid[1]}")
        self.prev_cen_y_times.append(video_time)
        #print(f"Time_y: {self.prev_cen_y_times[-1]}")

        vel = np.round(self.calculate_velocity()[1], 2)

        if len(self.velocities_y) > 0 and self.velocities_y[-1] == vel:
            return

        #if len(self.velocities_y) > 0 and vel == self.velocities_y[-1]:
        #    return

        #print(f"VELOCITY_Y: {vel}")
        #print(f"TIME_Y: {video_time}")
        self.velocities_y.append(vel)
        self.velocities_times_y.append(video_time)

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
        self.centroid = [int(min_x + (max_x - min_x)/2) + 1, 
                         int(min_y + (max_y - min_y)/2) + 1]
        
        # Calculate area of bbox
        self.area = self.bbox[2] * self.bbox[3]

class Tracker:
    def __init__(self, velocity_buffer=10, max_disappeared=10, max_distance=100, merge_threshold=5):
        self.next_id = 0
        self.objects = {}  # id -> Object
        self.disappeared = {}  # id -> frames_missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
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
                self.objects[self.next_id] = Object(centroid, 0, prev_cen_buf_x=deque(maxlen=self.velocity_buffer), prev_cen_buf_y=deque(maxlen=self.velocity_buffer))
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
                    self.objects[self.next_id] = Object(centroids[j], 0, 0, prev_cen_buf=deque(maxlen=self.velocity_buffer))
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
        
        # Prepare output
        results = []
        for obj_id, obj in self.objects.items():
            for region in regions:
                if region.centroid == obj.centroid:
                    obj.update_velocity_x()
                    obj.update_velocity_y()
                    results.append({
                        'id': obj_id,
                        'bbox': region.bbox,
                        'centroid': obj.centroid,
                        'area': region.area,
                        'velocity': obj.get_last_velocity(),
                        'acceleration': obj.calculate_acceleration()
                    })
                    obj.radius = math.sqrt(region.area) / 2
                    break

        return results

class MotionDetector:
    def __init__(self, min_area=500, threshold=25, noise_threshold=0.1, tracker=Tracker(50, 10, 10, 100), skip_preprocessing=True, cluster_leak=0, cluster_merge=0, shot_ball=Ball([0, 0], 5*5*math.pi, 1000, color=[0, 255, 255]), shoot_after_sec=2):
        self.min_area = min_area
        self.threshold = threshold
        self.tracker = tracker
        self.noise_threshold = noise_threshold
        self.shot_ball = shot_ball
        self.shoot_after_sec = shoot_after_sec

        self.skip_preprocessing = skip_preprocessing
        self.cluster_leak = cluster_leak
        self.cluster_merge = cluster_merge

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
            contentgancies_stack = [self.cluster_leak]
            # Grow region using stack-based flood fill
            while stack:
                y, x = stack.pop()
                contentgancy = contentgancies_stack.pop()

                if visited[y, x]:
                    continue
                    
                visited[y, x] = True

                if(image[y, x] > 0):
                    region.append([y, x])

                # Check all neighbors efficiently
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx

                    if(0 <= ny < height and 0 <= nx < width and 
                        image[ny, nx] == 0):
                        contentgancy-=1

                    # Bounds check and validation in single condition
                    if (0 <= ny < height and 0 <= nx < width and 
                        (image[ny, nx] > 0 or contentgancy > 0) and not visited[ny, nx]):
                        stack.append((ny, nx))
                        contentgancies_stack.append(contentgancy)

            # Only add regions that meet the minimum area requirement
            reg = Region(np.asarray(region))
            reg_w, reg_h = reg.bbox[2], reg.bbox[3]
            
            current_merged = False
            for other_region in regions:
                r1, r2 = np.sqrt(reg.area / np.pi), np.sqrt(other_region.area / np.pi)
                if (euclidean_distance(reg.centroid, other_region.centroid) < r1 + r2 + self.cluster_merge):
                    region_union = np.concatenate([reg.points, other_region.points], axis=0)
                    if (r1 > r2):
                        reg = Region(region_union)
                        regions.remove(other_region)
                    else:
                        other_region = Region(region_union)
                        current_merged = True

            is_square = reg_w != 0 and reg_h != 0 and 0.9 < (reg_w/reg_h) < 1.1

            if not current_merged and reg.area >= self.min_area and is_square:
                regions.append(reg)

            #print([start_x, start_y])

        return regions

    def preprocess_frame(self, frame):
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        edge_kernel = np.array([ 
                                [1,  1, 1],
                                [1, -9, 1],
                                [1,  1, 1]
                               ])

        edges = cv2.filter2D(gray, -1, edge_kernel)
        processed = edges
        cv2.imwrite("Preprocessed.png", processed)
        return processed

    def process_frame(self, frame):
        b, g, r = cv2.split(frame)
            
        if not self.skip_preprocessing:
            r = cv2.bitwise_or(self.preprocess_frame(r), self.preprocess_frame(~r))
            g = cv2.bitwise_or(self.preprocess_frame(g), self.preprocess_frame(~g))
            b = cv2.bitwise_or(self.preprocess_frame(b), self.preprocess_frame(~b))

        result = []
        if len(r) != 0:
            result = cv2.bitwise_or(cv2.bitwise_or(np.array(b), np.array(g)), np.array(r))
        
        proccessed = np.array(result)
            
        # Threshold
        _, thresh = cv2.threshold(proccessed, self.threshold, 255, cv2.THRESH_BINARY)

        regions = []
        if thresh is not None:
            regions = self.find_regions(thresh)
        
        return self.tracker.update(regions)
    
    def process_video(self, video_path, output_path=None, target_fps=-1):
        cap = cv2.VideoCapture(video_path)
        frame_size = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
                
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}")
        frametime = 1 / target_fps if target_fps != -1 else 1/fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count/fps

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 60, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        traj_pos = []
        traj_vel = []

        is_shot = False
        estimated_mass = 100
        target_hit = False
        global log

        shot_error_done = False
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            moving_objects = self.process_frame(frame)

            global video_time
            video_time += 1/fps
            
            # Draw results
            for obj in moving_objects:
                bbox = np.array(obj['bbox'])
                [x, y, w, h] = np.frompyfunc(int, 1, 1)(bbox)
                centroid = np.array(obj['centroid'])
                centroid = [int(centroid[0]), int(centroid[1])]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {obj['id']} bbox: {w}x{h}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Velocity: {(magnitude(obj['velocity'])):.2f} px/s", (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if(video_time < self.shoot_after_sec):
                    acc = np.array(obj['acceleration'], dtype=float)

                    mass = 0
                    
                    if(acc[0] != 0):
                        mass = -(world.C_d * magnitude(obj['velocity']) * obj['velocity'][0])/acc[0]
                    
                    if np.isnan(mass) or mass < 100:
                        mass = 100
                    
                    world.update_gravity(acc[1] + (world.C_d / mass) * magnitude(obj['velocity']) * obj['velocity'][1])
                    
                    if(world.get_avg_gravity() != 0 or acc[1] != 0):
                        mass = (mass + (-(world.get_avg_gravity() + world.C_d * magnitude(obj['velocity']) * obj['velocity'][1])/(acc[1]+world.get_avg_gravity())))/2
                    
                    if np.isnan(mass) or mass < 100:
                        mass = 100
                    estimated_mass = mass

                    traj_pos, _, _, _ = world.estimation_method(np.array(obj['centroid']), np.array(obj['velocity']), mass, frametime, 1)
                    traj_pos = [np.clip(pos, [0, 0], [frame_size[1] * 10, frame_size[0] * 10]) for pos in traj_pos]

                elif (not is_shot and len(traj_pos) != 0):
                    poses, vels = shooting_method(self.shot_ball, Ball(traj_pos[-1], 0, 0, color=[0, 0, 0]), frametime)
                    self.shot_ball.velocity = vels[0]
                    is_shot = True

                for pos in traj_pos:
                    if(np.any(np.isinf(pos)) or np.any(np.isnan(pos))):
                        continue
            
                    pos = [int(pos[0]), int(pos[1])]
                    cv2.circle(frame, pos, 2, (255, 255, 0), -1)

                if (is_shot and not target_hit and self.shot_ball.detect_collision(Ball(centroid, obj['area'], 0, color=[]))):
                    target_hit = True
                    print("Target hit!")
                if(video_time >= self.shoot_after_sec + 1 and not shot_error_done):
                    log+=f"Target hit error: {euclidean_distance(self.shot_ball.center, centroid)}\n"
                    shot_error_done=True
            if(is_shot):
                self.shot_ball.center = self.shot_ball.center + self.shot_ball.velocity * frametime
                self.shot_ball.velocity = self.shot_ball.velocity + (np.array([0, world.get_avg_gravity()])+self.shot_ball.calculate_drag()) * frametime
                cv2.circle(frame, np.array(self.shot_ball.center, dtype=int), 4, (0, 255, 0), -1)
                cv2.circle(frame, np.array(traj_pos[-1], dtype=int), 4, (255, 0, 255), -1)

            frame_end = time.time()
            ellapsed = frame_end - frame_start

            if(ellapsed < frametime):
                time.sleep(frametime - ellapsed)
                ellapsed = frametime
            
            if output_path:
                out.write(frame)

            cv2.imshow('Motion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        file_name = os.path.splitext(os.path.basename(output_path))[0] + ".txt"
        final_log = f"WORLD:\n    C_d={world.C_d}\n   esitmated gravity={world.get_avg_gravity()}\n   estimated_mass={estimated_mass}\nSHOOTING_METHOD:\n   shot_after={self.shoot_after_sec} sec\nTARGET HIT:    {target_hit}"
        log+=final_log
        with open(file_name, 'w') as file:
            file.write(log)

#####################################

if __name__ == "__main__":
    world.offset_threshold_x=5 # minimum offset between two tracked position points
    world.offset_threshold_y=5 #######################=SAME=#######################
    world.acceleration_step=5 # offset between two velocities used for acceleration estimation

    world.C_d=0.47/100 # drag constant of a sphere
    shot_ball = Ball([0, 0], 5*5*np.pi, 1000, [255, 255, 100])

    detector = MotionDetector(min_area=40, tracker=Tracker(100, 10, 25, 100), threshold=150, shot_ball=shot_ball, shoot_after_sec=2)
    video_path = "Videos/ball_interceptee1.mp4"
    detector.process_video(video_path, f"{os.path.splitext(video_path)[0]}_intercept_sim.mp4")