import numpy as np
import pygame
import sys
import cv2
import math

# Physics constants
MASS = 5  # kg
BALL_RADIUS = 15  # pixels
SIMULATION_TIME = 6
WORLD_TILT = np.array([-45, 75])

C_d = 0.47 / 100
G = 9.81*100/4
MASS = 1000

def magnitude(xy):
    return math.sqrt(xy[0]**2 + xy[1]**2)

def update_physics(position, velocity, dt):
    """Update position and velocity based on forces"""
    
    # Calculate acceleration (F = ma -> a = F/m)
    acceleration = np.array([  -(C_d/MASS)*velocity[0]*magnitude(velocity),
                             -G-(C_d/MASS)*velocity[1]*magnitude(velocity)], dtype=float)
    
    # Update velocity (v = v0 + at)
    new_velocity = velocity + acceleration * dt
    #print(new_velocity)
    # Update position (x = x0 + vt)
    new_position = np.array([0, 0], dtype=float)
    new_position[0] = position[0] + velocity[0] * dt
    new_position[1] = position[1] - velocity[1] * dt
    
    return new_position, new_velocity

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

    file_name = 'ball_interceptee0'

    video_out = cv2.VideoWriter(f'{file_name}.mp4', fourcc, 60.0, (width, height))

    # Initial conditions
    position = np.array([50, 150], dtype=float)  # pixels
    velocity = np.array([200, 100], dtype=float)  # px/s

    ### for generating text files
    #with open(file_name+".txt", 'w') as file:
    #    file.write(f"WORLD:\n   C_d={C_d}\n    gravity={G}\n    mass={MASS}\n"
    #    + f"INITIAL CONDITIONS:\n   position={position}\n   velocity={velocity}")

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

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw ball
        pos_pixels = position.astype(int)
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

    # Clean up
    video_out.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()