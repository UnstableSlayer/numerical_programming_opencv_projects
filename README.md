# Numerical Programming — OpenCV Projects

A collection of Python + OpenCV projects exploring object detection, physics simulation, and numerical methods for trajectory prediction and interception.

---

## Repository Structure

```
.
├── object_and_traffic_tracking/    # Object detection and velocity estimation
│   └── CP1.py
├── ode_estimation/                 # Physical parameter estimation via ODEs
│   ├── CP2.py
│   └── ball.py
├── hit_fixed_ball_targets/         # Final Exam Problem 1
│   ├── Final1.py
│   ├── Images/
│   └── Results/
├── hit_moving_ball_targets/        # Final Exam Problem 2
│   ├── Final2.py
│   ├── ball_simple.py              # Video generator
│   ├── Videos/
│   └── Results/
│       ├── Euler/
│       └── RK4/
```

---

## Projects

### object_and_traffic_tracking — Edge Detection, Object Detection & Movement Speed

Implements from-scratch moving object detection and velocity estimation from video using OpenCV, without any ML libraries.

**Pipeline:**
1. Extract frames and split into RGB channels (avoids information loss from grayscale conversion)
2. Preprocess frames using an edge-detection kernel and frame differencing to isolate moving regions
3. Segment the result using a **Region Growing** algorithm — a simple region-based clustering method that visits each pixel's neighbours and groups them into regions corresponding to moving objects
4. Calculate centroids of regions to track objects across frames
5. Estimate velocity from centroid displacement over time
6. Scale images using interpolation algorithms for faster computation

**Tested on:**
- Three-body problem simulation (works well; occasionally confuses ball identities when they overlap)
- Double pendulum simulation (no issues with identification or distinction)
- Real-world traffic footage (stable tracking; minor issues with shadows, black vehicles, and long vehicles)

**Known limitations:** Objects overlapping cause temporary ID confusion (tracking resumes after separation). Camera shake generates false detections. Black objects carry little color information and may jitter.

---

### ode_estimation — Estimating Mass, Velocity & Drag Force using ODEs

Extends the object detection code to estimate physical properties of a sliding disk from video using the **Runge-Kutta 4th order (RK4)** method.

**Physical model:** A 2D disk sliding on a surface with rigid walls (top-view), subject to drag force only (no gravity). Collisions with walls use restitution coefficient `e = 1`.

**Equations of motion:**

```
dr/dt = v
dv/dt = -(Cd/m) * v
```

Drag force: `Fd = 0.5 * Cd * rho * A * |v|^2 * v_hat`

**RK4 integration** computes intermediate slopes `k1`–`k4` at each timestep, giving O(dt^4) local truncation error — significantly more accurate than Euler's method.

**Estimation results on provided videos:**

| Video | Real velocity | Estimated velocity | Real drag | Estimated drag |
|---|---|---|---|---|
| ball_simulation_0 | 1.454 | 1.573 | 0.0544 | 0.0536 |
| ball_simulation_1 | 2.817 | 2.917 | 0.183 | 0.184 |
| ball_simulation_gravity_fail | 9.505 | 9.470 | 0.883 | 1.942 |

Velocity and drag are estimated accurately for simple videos. Mass cannot be reliably estimated due to pixel-level aliasing causing unstable acceleration readings. The gravity case fails for drag estimation since the model does not account for gravity.

---

### hit_fixed_ball_targets — Hit a Fixed Ball Target

Detects stationary balls in an image and simulates a projectile launched to hit each one, accounting for gravity and air resistance. Builds on the object detection from `object_and_traffic_tracking` and the ODE integration from `ode_estimation`.

**Objectives:**
- Detect static balls via computer vision (edge detection → thresholding → connected region filtering)
- Compute optimal initial velocity to reach each target using a **shooting method**
- Simulate trajectories with both Euler and RK4 integrators
- Visualize all trajectories on a single figure

**Equations of motion (with gravity):**

```
dx/dt  = vx          dy/dt  = vy
dvx/dt = -(Cd/m)*vx  dvy/dt = -g - (Cd/m)*vy
```

**Adaptive step size control** is applied to both methods: step halves when error exceeds tolerance, doubles when error drops below tolerance/10.

**Stability bounds** for the system (Cd=38, m=5):

| Method | Max stable dt |
|---|---|
| Euler | < 0.263 s |
| RK4   | < 0.733 s |

The chosen initial step of 0.001 s keeps both methods comfortably stable.

**To reproduce results:**
```python
world.g   = -12.61   # gravity
world.C_d = 38       # drag coefficient

detector = StaticBallDetector(min_area=400, threshold=50, skip_preprocessing=True)
detector.process_image("Images/random_balls2.jpg",
                       shot_pos=[7, 3], shot_radius=5, shot_mass=5)
```

For complex backgrounds, disable `skip_preprocessing` and enable `cluster_leak` / `cluster_merge`.

Output files are written to `Results/`: detection image, edge image, simulation figure, and a simulation log.

---

### hit_moving_ball_targets — Intercept a Moving Ball

Tracks a moving target ball in video and launches a projectile to intercept it in real time, estimating gravity and mass from the observed motion before shooting. Builds on all three preceding projects.

**Challenge:** The shot must be calculated using only the target's observed history — the algorithm watches the ball for a set number of seconds, estimates its physical parameters, predicts where it will be, then fires.

**System components:**
- `WorldProperties` — physical constants
- `Ball` — physics integration and collision detection
- `Object` — per-object tracking and velocity calculation
- `Tracker` — multi-object tracking
- `MotionDetector` — video processing and simulation coordination

**Shooting method:** Iteratively adjusts the initial velocity vector of the projectile, simulating the trajectory at each iteration and correcting based on the distance error from the predicted target position, until convergence.

**Test results:**

| Scenario | Method | Hit | Notes |
|---|---|---|---|
| ball_interceptee0 | Euler & RK4 | Yes | Clean interception |
| ball_interceptee1 | Euler & RK4 | Yes | Small residual hit error (~17 px) |
| ball_interceptee1 (fail) | Euler & RK4 | No | Too little estimation time; gravity underestimated |
| ball_interceptee2 (ascending) | Euler & RK4 | Yes | Hit error < 1.1 px |
| ball_interceptee2 (descending) | Euler & RK4 | Yes | Hit error ~6.9 px |
| ball_interceptee2 (stop & reverse) | Euler & RK4 | No | Cannot predict a ball that stops and re-accelerates mid-flight |

**Key failure modes:**
- Insufficient observation time → inaccurate gravity estimate → miss
- A target that decelerates to a stop and then re-accelerates cannot be predicted from instantaneous velocity/acceleration history alone

For background handling, use the same approach as `hit_fixed_ball_targets`: disable `skip_preprocessing`, enable `cluster_leak` and `cluster_merge`.

---

## Numerical Methods Summary

| Method | Truncation Error | Stable step (Cd=38, m=5) | Notes |
|---|---|---|---|
| Euler | O(dt) | < 0.263 s | Conditionally stable; simple, fast |
| RK4   | O(dt^4) | < 0.733 s | Much more accurate; allows larger steps |

Both methods use adaptive step size control. RK4 is preferred for accuracy-critical simulations.

---

## Dependencies

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib

---

## Author

Demetre Lataria
