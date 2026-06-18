import numpy as np
import cv2
from collections import deque
from typing import List, Dict, Tuple
import time
import AP3
import math

WIDTH = 0
HEIGHT = 0
video_time = 0
video_fps = 0
video_length = 0

density = 1
max_iter = 1000

def round_cause_python(x):
    return math.ceil(x*100)/100

def euclidean_distance(v1,v2):
    return math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(v1, v2)))

def magnitude(xy):
    return math.sqrt(xy[0]**2 + xy[1]**2)

class Object:
    def __init__(self, centroid, density, radius, pixels_per_meter, prev_cen_buf=deque(maxlen=1000)):
        self.centroid = centroid
        self.density = density
        self.radius = radius
        self.prev_cen_buf = prev_cen_buf
        self.prev_cen_times = deque(maxlen=prev_cen_buf.maxlen)
        self.creation_time = video_time

        self.pixels_per_meter = pixels_per_meter
        
        self.velocities = deque(maxlen=prev_cen_buf.maxlen - 1)
        self.accelerations = deque(maxlen=prev_cen_buf.maxlen - 2)

        self.positions = []

        self.iter = 0

    def get_trajectory(self):
        #print(f"POS: {self.centroid}")
        #print(f"VEL: {self.calculate_last_velocity()}")
        #print(f"ACCL: {self.calculate_last_acceleration()}")

        if len(self.prev_cen_buf) < 2:
            return [], []

        global drag, gravity, density, video_fps

        global max_iter
        
        vels = []
        if self.iter < max_iter:
            self.positions, vels = simulate_ball_motion_rk4(self.radius / self.pixels_per_meter, density,
                                    np.array(self.prev_cen_buf[-1]) / self.pixels_per_meter, self.calculate_last_velocity(), self.calculate_last_acceleration(),
                                    1/(video_fps), WIDTH/self.pixels_per_meter, HEIGHT / self.pixels_per_meter, video_length) 
            self.iter+=1
        return self.positions * self.pixels_per_meter, vels

    def calculate_last_velocity(self):
        if len(self.prev_cen_buf) > 1:
            #print(f"[ {np.divide(self.prev_cen_buf[-1], self.pixels_per_meter)} ]")
            return (np.subtract(np.divide(self.prev_cen_buf[-1], self.pixels_per_meter), np.divide(self.prev_cen_buf[-2], self.pixels_per_meter))) / 0.016
        
        return [0, 0]

    def calculate_last_acceleration(self):
        if len(self.prev_cen_buf) > 2:
            a = (np.subtract(self.prev_cen_buf[-2], self.prev_cen_buf[-3]) / self.pixels_per_meter) / 0.016
            b = (np.subtract(self.prev_cen_buf[-1], self.prev_cen_buf[-2]) / self.pixels_per_meter) / 0.016
            return np.subtract(b, a) / 0.016
        
        return [0, 0]

    def update_velocity(self):
        self.prev_cen_buf.append(self.centroid)
        self.prev_cen_times.append(video_time)

        vel = self.calculate_last_velocity()
        if magnitude(vel) > 0:
            self.velocities.append(vel)

        acc = self.calculate_last_acceleration()
        if magnitude(acc) > 0 and len(self.accelerations) > 0 and magnitude(acc) < np.max([magnitude(a) for a in self.accelerations]):
            self.accelerations.append(acc)

    def get_estimations(self):
        est_velocity = np.average([magnitude(v) for v in self.velocities])
        est_drag = 0.5 * 0.5 * 1.225 * (np.pi * (self.radius / self.pixels_per_meter)**2) * est_velocity**2
        avg_acc = np.average([magnitude(a) for a in self.accelerations]) if len(self.accelerations) > 0 else 0
        #print(f"AVERAGE: {avg_acc}")
        est_mass = est_drag / avg_acc if avg_acc != 0 else -1
        return est_mass, est_velocity, est_drag

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
                self.objects[self.next_id] = Object(centroid, 0, 0, self.pixels_per_meter, prev_cen_buf=deque(maxlen=self.velocity_buffer))
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
                    self.objects[self.next_id] = Object(centroids[j], 0, 0, self.pixels_per_meter, prev_cen_buf=deque(maxlen=self.velocity_buffer))
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
        
        # Prepare output
        results = []
        for obj_id, obj in self.objects.items():
            for region in regions:
                if region.centroid == obj.centroid:
                    obj.update_velocity()
                    mass, vel, drag = obj.get_estimations()
                    results.append({
                        'id': obj_id,
                        'bbox': region.bbox,
                        'centroid': obj.centroid,
                        'area': region.area,
                        'velocity': obj.calculate_last_velocity(),
                        'trajectory': obj.get_trajectory(),
                        'estimations': f"Last Estimated:\n  mass: {mass}\n  velocity: {vel}\n  drag: {drag}"
                    })
                    obj.radius = math.sqrt(region.area) / 2
                    break

        return results

def simulate_ball_motion_rk4(r, e, pos, vel, accel, dt, W, H, t_max):
    [x, y] = pos
    [vx, vy] = vel
    [ax, ay] = accel

    n_steps = int(t_max / dt)

    area = np.pi * r**2
    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))

    # Store the initial position and velocity
    positions[0] = [x, y]
    velocities[0] = [vx, vy]

    C_d = 0.5 # Drag coefficient (constant, depends on material and surface roughness of the ball which can not be identified from video alone)
    rho = 1.225 # Air density
    
    F_d = 0.5 * C_d * rho * area * magnitude(vel)**2

    mass = F_d / magnitude(accel) if magnitude(accel) != 0 else 0.01

    #print(f'Estimated mass: {mass}')

    # Define the ODE system for Runge-Kutta
    def derivatives(pos, vel):
        x, y = pos
        vx, vy = vel
        ax = -C_d * vx
        ay = -C_d * vy
        return np.array([vx, vy]), np.array([ax, ay])

    # Iterate over time steps
    for i in range(1, n_steps):
        # Runge-Kutta 4th order method
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
        vx += (dt / 6) * (k1_acc[0] + 2 * k2_acc[0] + 2 * k3_acc[0] + k4_acc[0])
        vy += (dt / 6) * (k1_acc[1] + 2 * k2_acc[1] + 2 * k3_acc[1] + k4_acc[1])
        x += (dt / 6) * (k1_vel[0] + 2 * k2_vel[0] + 2 * k3_vel[0] + k4_vel[0])
        y += (dt / 6) * (k1_vel[1] + 2 * k2_vel[1] + 2 * k3_vel[1] + k4_vel[1])

        # Handle collisions with the box boundaries (elastic collisions with restitution)
        if x - r <= 0 or x + r >= W:
            vx = -e * vx
            x = -x + 2 * r if x - r <= 0 else 2 * W - 2 * r - x

        if y - r <= 0 or y + r >= H:
            vy = -e * vy  # Reverse velocity in y-direction and apply restitution
            y = -y + 2 * r if y - r <= 0 else 2 * H - 2 * r - y

        # Store updated positions and velocities
        positions[i] = [x, y]
        velocities[i] = [vx, vy]

    return positions, velocities

class MotionDetector:
    def __init__(self, min_area=500, threshold=25, noise_threshold=0.1, tracker=Tracker(50, 10, 10, 100), image_scale=1, image_smear=1, history_frames=1, skip_preproccessing=False):
        self.min_area = min_area
        self.threshold = threshold
        self.tracker = tracker
        tracker.pixels_per_meter *= image_scale
        self.image_scale = image_scale
        self.image_smear = image_smear
        self.history_frames = history_frames
        self.prev_frames = []
        self.noise_threshold = noise_threshold
        self.skip_preproccessing = skip_preproccessing

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
        
        proccessed = cv2.bitwise_and(proccessed, edges)
        proccessed = cv2.filter2D(proccessed, -1, self.gaussian_kernel(3, 3))
        proccessed = cv2.filter2D(proccessed, -1, smear_kernel)
        proccessed = cv2.filter2D(proccessed, -1, leak_kernel)

        return proccessed

    def process_frame(self, frame):
        b, g, r = cv2.split(frame)
        
        if not self.skip_preproccessing:
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
            
        # Threshold
        _, thresh = cv2.threshold(proccessed, self.threshold, 255, cv2.THRESH_BINARY)

        regions = []
        if thresh is not None:
            regions = self.find_regions(thresh)
        
        return self.tracker.update(regions)
    
    def process_video(self, video_path, output_path=None, target_fps=-1):
        cap = cv2.VideoCapture(video_path)
        old_size = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
        new_size = np.multiply(old_size, self.image_scale)
        
        global WIDTH, HEIGHT
        WIDTH = new_size[1]
        HEIGHT = new_size[0]
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        global video_fps, video_length
        video_fps = fps
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / video_fps
        frametime = 1 / target_fps if target_fps != -1 else 1/fps

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 60, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        last_pos = []
        while cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = AP3.nearest_neighbor_interpolation(frame, new_size[0], new_size[1])

            # Process frame
            moving_objects = self.process_frame(resized_frame)

            frame_end = time.time()
            ellapsed = frame_end - frame_start

            global video_time
            video_time += 1/fps
            
            if(ellapsed < frametime):
                time.sleep(frametime - ellapsed)

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
                cv2.putText(frame, f"Velocity: {(magnitude(obj['velocity'])):.2f} m/s", (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw Trajectory
                positions, velocities = obj['trajectory']
                accs = []
                for i in range(1, len(velocities)):
                    accs.append((velocities[i] - velocities[i-1])/0.016)

                est_velocity = np.average([magnitude(v) for v in velocities])
                est_drag = 0.5 * 0.5 * 1.225 * (np.pi * ((math.sqrt(obj['area']) / 2) / self.tracker.pixels_per_meter)**2) * est_velocity**2
                avg_acc = np.average([magnitude(a) for a in accs]) if len(accs) > 0 else 0
                print(f"AVG: {avg_acc}")
                est_mass = est_drag / avg_acc if avg_acc != 0 else -1

                print(f"Estimated:\n  mass: {est_mass}\n  velocity: {est_velocity}\n  drag: {est_drag}")


                for position in positions:
                    position = np.clip(position, [0, 0], [WIDTH, HEIGHT])
                    cv2.circle(frame, [int(position[0] / self.image_scale), int(position[1] / self.image_scale)], 4, (255, 0, 0), -1)
            
                if (video_length - video_time < 1/fps):
                    print(obj['estimations'])

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
    detector = MotionDetector(min_area=40, tracker=Tracker(100, 10, 25, 100), skip_preproccessing=True, image_scale=1, threshold=150)
    video_path = "ball_simulation_gravity_fail.mp4"
    detector.process_video(video_path, f"CP2_{video_path}")