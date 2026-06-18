import numpy as np
import pygame
import sys
import cv2
import math

# Physics constants
MASS = 5  # kg
DRAG_COEFFICIENT = 0.5  # Sphere drag coefficient
AIR_DENSITY = 1.225  # kg/m^3
BALL_RADIUS = 15  # pixels
PIXEL_TO_METER = 0.01  # 1 pixel = 1 cm
METER_TO_PIXEL = 100  # 1 meter = 100 pixels

SIMULATION_TIME = 5

DRAGS = []

def magnitude(xy):
    return math.sqrt(xy[0]**2 + xy[1]**2)

def calculate_drag_force(velocity, area):
    """Calculate drag force: F = -1/2 * ρ * v^2 * Cd * A * v̂"""
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude != 0:
        velocity_unit = velocity / velocity_magnitude
        drag_magnitude = 0.5 * AIR_DENSITY * velocity_magnitude**2 * DRAG_COEFFICIENT * area

        DRAGS.append(drag_magnitude)

        return -drag_magnitude * velocity_unit
    return np.zeros(2)

def update_physics(position, velocity, dt):
    """Update position and velocity based on forces"""
    # Calculate cross-sectional area
    area = np.pi * (BALL_RADIUS * PIXEL_TO_METER) ** 2
    
    # Calculate forces
    force = calculate_drag_force(velocity, area)
    # Calculate acceleration (F = ma -> a = F/m)
    acceleration = force / MASS
    acceleration[1] += 9.98 #In case of gravity fail testing
    
    #print(np.linalg.norm(force))
    #print(f"MASS: {MASS} == {np.linalg.norm(force) / np.linalg.norm(acceleration)}")
    #acceleration = force / mass
    #             = -drag_magnitude * velocity_unit / mass
    #             = -(0.5 * AIR_DENSITY * velocity_magnitude ** 2 * C_d * area) * vel_unit / mass
    #mass * accel = -0.5 * AIR_DENSITY * vel_mag**2 * C_d * area * vel_unit
    #mass * accel / (-0.5 * AIR_DENSITY * vel_mag**2 * area * vel_unit) = C_d
    
    # Update velocity (v = v0 + at)
    new_velocity = velocity + acceleration * dt
    
    # Update position (x = x0 + vt)
    new_position = position + velocity * dt
    
    return new_position, new_velocity

def handle_boundaries(position, velocity, screen_width, screen_height):
    """Handle collisions with screen boundaries"""
    screen_bounds = np.array([screen_width * PIXEL_TO_METER, 
                            screen_height * PIXEL_TO_METER])
    
    [x, y] = position
    [vx, vy] = velocity
    [W, H] = screen_bounds
    r = BALL_RADIUS * PIXEL_TO_METER

    if x - r <= 0 or x + r >= W:
        vx = -vx
        x = -x + 2 * r if x - r <= 0 else 2 * W - 2 * r - x

    if y - r <= 0 or y + r >= H:
        vy = -vy  # Reverse velocity in y-direction and apply restitution
        y = -y + 2 * r if y - r <= 0 else 2 * H - 2 * r - y
            
    return np.asarray([x, y]), np.asarray([vx, vy])

def pygame_surface_to_cv2_image(surface):
    """Convert Pygame surface to CV2 image format"""
    view = pygame.surfarray.pixels3d(surface)
    view = view.transpose([1, 0, 2])  # Transpose to get correct orientation
    return cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

def main():
    # Initialize Pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("2D Ball Physics Simulation")

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('ball_simulation_gravity_fail.mp4', fourcc, 60.0, (width, height))

    # Initial conditions
    position = np.array([2.0, 1.0], dtype=float)  # meters
    velocity = np.array([3, 0], dtype=float)  # m/s

    # Clock for controlling simulation speed
    clock = pygame.time.Clock()
    FPS = 60
    dt = 1/FPS

    # Simulation duration (10 seconds)
    current_frame = 0

    running = True
    while running and current_frame / FPS < SIMULATION_TIME:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update physics
        position, velocity = update_physics(position, velocity, dt)
        position, velocity = handle_boundaries(position, velocity, width, height)

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw ball
        pos_pixels = (position * METER_TO_PIXEL).astype(int)
        print(pos_pixels)
        pygame.draw.circle(screen, (255, 0, 0), pos_pixels, BALL_RADIUS)
        pygame.draw.circle(screen, (0, 0, 255), pos_pixels, 2)

        # Update display
        pygame.display.flip()

        # Capture frame for video
        frame = pygame_surface_to_cv2_image(screen)
        video_out.write(frame)
        
        current_frame += 1

        # Control frame rate
        clock.tick(FPS)

    print(f"Real:\n  mass: {MASS}\n  velocity: {magnitude(velocity)}\n  drag: {np.average(DRAGS)}")

    # Clean up
    video_out.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()