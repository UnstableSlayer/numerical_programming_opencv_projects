import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import math

import os

sim_log = ""

def euclidean_distance(v1,v2):
    return math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(v1, v2)))

def magnitude(v):
    return math.sqrt(sum(x**2 for x in v))

def calculate_drag(mass, velocity):
    return np.array((world.C_d/mass) * velocity, dtype=np.float128) 

def generate_distinct_colors(n):
    cmap = plt.get_cmap('tab10', n)
    colors = [cmap(i) for i in range(n)]
    colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
    return colors

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
        vy1 = vy - (world.g + f_d1[1]) * dt
        
        # Two half steps
        dt_half = dt / 2
        xh, yh = x + vx * dt_half, y + vy * dt_half
        f_dh = calculate_drag(mass, np.array([vx, vy]))
        vxh = vx - f_dh[0] * dt_half
        vyh = vy - (world.g + f_dh[1]) * dt_half
        
        x2, y2 = xh + vxh * dt_half, yh + vyh * dt_half
        f_d2 = calculate_drag(mass, np.array([vxh, vyh]))
        vx2 = vxh - f_d2[0] * dt_half
        vy2 = vyh - (world.g + f_d2[1]) * dt_half
        
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
        ay = -world.g - f_d[1]
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

    global sim_log
    init_log = f"Initializing shooting method: {f1_0}, {f1_1}"
    sim_log += init_log+'\n'  
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
    sim_log += end_log
    
    return poses, vels

class WorldProperties:
    def __init__(self):
        self.g = 9.81     #px/s^2 100 px == 1 m
        self.C_d = 0.47#*100
        self.meter_to_pixel = 100
        self.estimation_method = simulate_ball_motion_rk4
world = WorldProperties()

class Ball:
    def __init__(self, center, radius, mass, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = color
        self.mass = mass
        
        self.velocity = np.array([0, 0])

        self.animated_ball = self.to_matplotlib_ball()
        self.positions = []
        self.velocities = []

    def calculate_drag(self):
        return calculate_drag(self.mass, self.velocity)

    def detect_collision(self, other):
        return euclidean_distance(self.center, other.center) < self.radius + other.radius

    def to_matplotlib_ball(self):
        return plt.Circle((float(self.center[0]), float(self.center[1])), self.radius, color=np.divide(self.color, 255), fill=True)

    def update_animated(self):
        self.animated_ball.center = (float(self.center[0]), float(self.center[1]))
        self.animated_ball.radius = self.radius
        self.animated_ball.color = self.color

    def update_ball(self, dt):
        self.center = self.center + self.velocity * dt
        self.velocity = self.velocity + (-np.array([0, world.g]) - self.calculate_drag())*dt

        return self.center

    def update_ball_est(self, t, dt):
        self.center = self.positions[int(((len(self.positions)-1)/(1/dt))*((t/dt)+1))]
        self.velocity = self.velocities[int(((len(self.positions)-1)/(1/dt))*((t/dt)+1))]
        return self.center

class BallSimulation:
    def __init__(self, balls, width, height, shot_pos=[0, 0], shot_radius=5, shot_mass=5, dt=0.01, sequencial=False):
        self.balls = balls
        self.width = width
        self.height = height

        self.shot_pos = shot_pos
        self.shot_balls = [[Ball(self.shot_pos, shot_radius/world.meter_to_pixel, shot_mass, [0, 0, 0]),
                            Ball(self.shot_pos, shot_radius/(world.meter_to_pixel*2), shot_mass, [0, 0, 0])] for _ in self.balls]

        self.dt = dt
        self.t = 0

        self.colors = generate_distinct_colors(len(balls))

        self.fig, self.ax = plt.subplots()
        self.patches = []
        self.tail_iter = 0

        self.hits = []
        self.sequencial = sequencial

    def simulate_balls(self):
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')

        title = f'Bouncing Ball Animation:\n        Gravity: {world.g:.5f}, C_d: {world.C_d:.5f},\n        Shot ball mass: {self.shot_balls[0][0].mass}, Shot ball radius: {self.shot_balls[0][0].radius}, Shot from pos: {self.shot_pos}, max_dt={self.dt}'
        global sim_log
        sim_log += title + '\n\n'
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
        ani = animation.FuncAnimation(self.fig, func=self.update, init_func=self.init, frames=np.linspace(0, 1, int(1/self.dt)+1), blit=True, interval=self.dt*1000, repeat=False)
       
        plt.show()

    def init(self):
        if self.patches:
            return self.patches

        for ball in self.balls:
            self.patches.append(self.ax.add_patch(ball.animated_ball))

        for i,[ball, particle] in enumerate(self.shot_balls):
            self.patches.append(self.ax.add_patch(ball.animated_ball))
            self.patches.append(self.ax.add_patch(particle.animated_ball))
            global sim_log
            ball_log = f"Ball with ID {i}: \n"
            print(ball_log)
            sim_log += ball_log
            particle.positions, particle.velocities = shooting_method(ball, self.balls[i], self.dt)
            particle.velocity = particle.velocities[0]

        return self.patches

    def update(self, t):   
        gravity = np.array([0, world.g], dtype=float)
        global sim_log
        dt = self.dt

        for i,[ball, particle] in enumerate(self.shot_balls):
            target_ball = self.balls[i]

            if self.sequencial:
                i = int(t)
                target_ball = self.balls[i]
                shot_ball = self.shot_balls[i][0]
                shot_part = self.shot_balls[i][1]
            else:
                shot_ball = ball
                shot_part = particle

            # Check collision
            if not shot_ball.detect_collision(target_ball):
                shot_ball.center = shot_part.center
            
            if not self.sequencial and t >= 1 or self.sequencial and t >= len(self.balls):
                shot_part.velocity = np.array([0, 0])
                if(i not in self.hits):
                    if shot_ball.detect_collision(target_ball):
                        sim_log+=f"Ball with ID {i} result: HIT, pos:{shot_part.center}, target: {target_ball.center}, error: {euclidean_distance(shot_part.center, target_ball.center)}\n"
                    else:
                        sim_log+=f"Ball with ID {i}, result: FAIL\n"
                    self.hits.append(i)

                if self.sequencial:
                    break
            else:
                shot_part.update_ball(self.dt)

            if(len(self.hits) == len(self.balls)):
                file_name = "Results/"+os.path.splitext(os.path.basename(image_path))[0] + suffix + "_sim_fig.png"
                plt.savefig(file_name)

            if(self.tail_iter % (int((1/dt)/30 if 30 < 1/dt else 1)) == 0):
                self.patches.append(self.ax.add_patch(plt.Circle((float(shot_ball.center[0]), float(shot_ball.center[1])),
                                shot_ball.radius / 2, color=np.divide(self.colors[i], 255), fill=True)))
                
            shot_ball.update_animated()
            shot_part.update_animated()

            if self.sequencial:
                break

        self.t = t
        self.tail_iter += 1
        return self.patches

class Region:
    def __init__(self, points):
        self.points = points
        if len(points) == 0:
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

class StaticBallDetector:
    def __init__(self, min_area=500, threshold=25, skip_preprocessing=True, cluster_leak=0, cluster_merge=0):
        self.min_area = min_area
        self.threshold = threshold

        self.cluster_leak = cluster_leak
        self.cluster_merge = cluster_merge

        self.skip_preprocessing = skip_preprocessing

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

    def regions_to_balls(self, regions, image):
        balls = []

        for reg in regions:
            height, width, _ = image.shape
            ball_center = [reg.centroid[0]/world.meter_to_pixel, (height - reg.centroid[1])/world.meter_to_pixel]
            ball_color = image[reg.centroid[1]][reg.centroid[0]]
            ball_color = [ball_color[2], ball_color[1], ball_color[0]]
            ball = Ball(ball_center, (reg.bbox[2]+reg.bbox[3])/(4*world.meter_to_pixel), 0, ball_color)
            balls.append(ball)
        
        return balls

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

        smear_kernel = np.array([
                                 [1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]
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
            result = cv2.bitwise_or(cv2.bitwise_or(b, g), r)
        
        proccessed = np.array(result)

        # Threshold
        _, thresh = cv2.threshold(proccessed, self.threshold, 255, cv2.THRESH_BINARY)

        file_name = "Results/"+os.path.splitext(os.path.basename(image_path))[0] + "_edge.png"
        cv2.imwrite(file_name, thresh)

        regions = []
        if thresh is not None:
            regions = self.find_regions(thresh)
        
        return regions

    def process_image(self, image_path, shot_pos=[0, 0], shot_radius=5, shot_mass=5, dt=0.001):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        frame_size = [np.shape(image)[0], np.shape(image)[1]]

        # Process frame
        regions = self.process_frame(image)
        balls = self.regions_to_balls(regions, image)

        frame = image

        # Draw results
        for i,reg in enumerate(regions):
            bbox = np.array(reg.bbox)
            [x, y, w, h] = np.frompyfunc(int, 1, 1)(bbox)
            centroid = np.array(reg.centroid)
            centroid = [int(centroid[0]), int(centroid[1])]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {i} bbox: {w}x{h}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        file_name = "Results/"+os.path.splitext(os.path.basename(image_path))[0] + "_detection.png"
        cv2.imwrite(file_name, frame)

        sim = BallSimulation(balls, frame_size[1]/world.meter_to_pixel, frame_size[0]/world.meter_to_pixel, shot_pos, shot_radius, shot_mass, dt)
        sim.simulate_balls()

        file_name = "Results/"+os.path.splitext(os.path.basename(image_path))[0] + ("_euler" if world.estimation_method == simulate_ball_motion_euler else "_rk4") + suffix + "_sim.txt"

        global sim_log
        print(sim_log)
        with open(file_name, 'w') as file:
            file.write(sim_log)

if __name__ == "__main__":
    #world.estimation_method = simulate_ball_motion_euler
    world.g = -12.61
    world.C_d = 38

    detector = StaticBallDetector(min_area=400, threshold=50, skip_preprocessing=True)
    image_path = "Images/random_balls2.jpg"
    suffix = ""
    detector.process_image(image_path, [7, 3], 5, 5)

    #detector = StaticBallDetector(min_area=400, threshold=50, skip_preprocessing=False, cluster_leak=500, cluster_merge=10)
    #image_path = "Images/background_test.png"
    #detector.process_image(image_path, [1, 3], 5, 5)

    #detector = StaticBallDetector(min_area=400, threshold=50, skip_preprocessing=False, cluster_leak=0, cluster_merge=0, shot_pos=[1, 3], shooting_method_h_step=1/world.meter_to_pixel)
    #image_path = ".jpg"
