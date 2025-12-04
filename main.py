import pygame
import random
import math
import sys
import cv2
import numpy as np
import degirum as dg

pygame.init()

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 600
EYE_RADIUS = 200
PUPIL_RADIUS = 50
PUPIL_RANGE = 120
BACKGROUND_COLOR = (30, 30, 30)
EYE_COLOR = (240, 240, 240)
IRIS_BASE_COLOR = (70, 100, 200)
PUPIL_COLOR = (20, 20, 20)
HIGHLIGHT_COLOR = (255, 255, 255)

ENABLE_MICRO_SACCADES = True
ENABLE_IRIS_GRADIENT = True

# --- SETUP ---
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

eye_center = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
pupil_offset = pygame.Vector2(0, 0)
velocity = pygame.Vector2(0, 0)
target = pygame.Vector2(eye_center)

# ============================
# 1. DeGirum & Camera Setup
# ============================
print(dg.get_supported_devices(dg.CLOUD))

zoo = dg.connect("127.0.0.1")

model_name = "yolov8n_relu6_face--640x640_quant_hailort_multidevice_1"
model = zoo.load_model(model_name)

cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Cannot open camera")
    pygame.quit()
    sys.exit()

model_width, model_height = 640, 640


# ============================
# 2. Utility Functions
# ============================
def set_target(position: pygame.Vector2):
    global target
    target = position


def smooth_damp(current, target, velocity, smooth_time, delta):
    omega = 2.0 / smooth_time
    x = omega * delta
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    temp = (velocity + omega * change) * delta
    new_velocity = (velocity - omega * temp) * exp
    new_value = target + (change + temp) * exp
    return new_value, new_velocity


def get_pupil_target():
    direction = target - eye_center
    if direction.length() == 0:
        return pygame.Vector2(0, 0)
    direction = direction.normalize()
    dist = min(PUPIL_RANGE, (target - eye_center).length() / 5)
    return direction * dist


def update_pupil(dt):
    global pupil_offset, velocity

    target_offset = get_pupil_target()

    if ENABLE_MICRO_SACCADES:
        jitter = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)) * 2
        target_offset += jitter

    pupil_offset, velocity = smooth_damp(
        pupil_offset, target_offset, velocity, smooth_time=0.25, delta=dt
    )


def draw_iris_gradient(center, base_color, radius):
    for i in range(radius, 0, -1):
        t = i / radius
        shade = (
            int(base_color[0] * t + 20),
            int(base_color[1] * t + 20),
            int(base_color[2] * t + 20),
        )
        pygame.draw.circle(screen, shade, center, i)


def draw_eye():
    screen.fill(BACKGROUND_COLOR)

    pygame.draw.circle(screen, EYE_COLOR, eye_center, EYE_RADIUS)

    iris_pos = eye_center + pupil_offset * 0.7
    if ENABLE_IRIS_GRADIENT:
        draw_iris_gradient(iris_pos, IRIS_BASE_COLOR, int(PUPIL_RADIUS * 1.8))
    else:
        pygame.draw.circle(screen, IRIS_BASE_COLOR, iris_pos, int(PUPIL_RADIUS * 1.8))

    pupil_pos = eye_center + pupil_offset
    pygame.draw.circle(screen, PUPIL_COLOR, pupil_pos, PUPIL_RADIUS)

    time = pygame.time.get_ticks() * 0.002
    highlight_offset = pygame.Vector2(math.sin(time) * 5, math.cos(time) * 5)
    pygame.draw.circle(
        screen,
        HIGHLIGHT_COLOR,
        pupil_pos + highlight_offset + pygame.Vector2(-10, -10),
        10,
    )

    pygame.display.flip()


# ============================
# 3. Face Detection Function
# ============================
def detect_face_center(frame):
    """
    Returns (x, y) of face center mapped to pygame screen.
    Also returns the frame with detection boxes for display.
    """

    global model_width, model_height

    orig_h, orig_w = frame.shape[:2]

    draw_frame = frame.copy()  # for OpenCV window

    # Letterbox
    scale = min(model_width / orig_w, model_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = (model_width - new_w) // 2
    pad_h = (model_height - new_h) // 2

    input_frame = cv2.copyMakeBorder(
        resized,
        pad_h, model_height - new_h - pad_h,
        pad_w, model_width - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Run inference
    results = model(input_frame)

    best_face = None
    best_score = 0

    for r in results.results:
        x1, y1, x2, y2 = r["bbox"]
        score = r["score"]

        if score < 0.4:
            continue

        # Undo padding + scaling
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Draw box on the OpenCV window frame
        cv2.rectangle(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Track the highest confidence face
        if score > best_score:
            best_score = score
            best_face = ((x1 + x2) / 2, (y1 + y2) / 2)

    # If no face
    if best_face is None:
        return None, draw_frame

    # Convert to pygame coordinates
    cam_x, cam_y = best_face
    scale_x = WIDTH / orig_w
    scale_y = HEIGHT / orig_h

    return pygame.Vector2(cam_x * scale_x, cam_y * scale_y), draw_frame


# ============================
# 4. Main Loop
# ============================
def main():
    global ENABLE_MICRO_SACCADES, ENABLE_IRIS_GRADIENT

    while True:
        dt = clock.tick(60) / 1000

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    ENABLE_MICRO_SACCADES = not ENABLE_MICRO_SACCADES
                elif event.key == pygame.K_2:
                    ENABLE_IRIS_GRADIENT = not ENABLE_IRIS_GRADIENT

        # Read camera
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1) # Flip Horizontally
        face_pos, debug_frame = detect_face_center(frame)

        # Set eye target
        if face_pos is None:
            set_target(eye_center)
        else:
            set_target(face_pos)

        # Update eye
        update_pupil(dt)
        draw_eye()

        # Show secondary window
        cv2.imshow("Face Tracking", debug_frame)

        # Allow exit via ESC in OpenCV window
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
