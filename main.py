import cv2
import mediapipe as mp
import pyautogui
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "hand_landmarker.task"

# Camera control area
FRAME_MARGIN_X = 80
FRAME_MARGIN_Y = 60

# Cursor smoothing
SMOOTHING = 0.45

# Pinch distance
PINCH_THRESHOLD = 0.045

# Right-click distance
RIGHT_CLICK_DISTANCE = 0.08

# Time between clicks
CLICK_COOLDOWN = 0.5

# Scroll sensitivity
SCROLL_SENSITIVITY = 10

# Minimum movement before scrolling
SCROLL_DEADZONE = 8

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


# ============================================================
# SCREEN
# ============================================================

screen_width, screen_height = pyautogui.size()

print("Screen:", screen_width, "x", screen_height)


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    detector.close()

    exit()


# ============================================================
# VARIABLES
# ============================================================

previous_mouse_x = screen_width // 2
previous_mouse_y = screen_height // 2

last_click_time = 0

# Pinch state
pinching = False

# Drag state
dragging = False

# Scroll state
scrolling = False

previous_scroll_y = None

# Gesture cooldown
last_right_click_time = 0


# ============================================================
# START MESSAGE
# ============================================================

print()
print("================================")
print("        AIR MOUSE STARTED")
print("================================")
print()
print("☝️  Index       = Move")
print("🤏  Pinch       = Left Click")
print("🤏  Hold Pinch = Drag")
print("✌️  Two Fingers = Right Click")
print("👆  Two Fingers + Move = Scroll")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape


    # --------------------------------------------------------
    # Draw control area
    # --------------------------------------------------------

    cv2.rectangle(
        frame,

        (FRAME_MARGIN_X, FRAME_MARGIN_Y),

        (
            frame_width - FRAME_MARGIN_X,
            frame_height - FRAME_MARGIN_Y
        ),

        (255, 255, 255),

        2
    )


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(time.time() * 1000)


    # --------------------------------------------------------
    # Detect hand
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # ====================================================
        # LANDMARKS
        # ====================================================

        wrist = hand[0]

        thumb_tip = hand[4]

        index_tip = hand[8]

        middle_tip = hand[12]

        ring_tip = hand[16]

        pinky_tip = hand[20]


        # ====================================================
        # PIXEL COORDINATES
        # ====================================================

        index_x = int(index_tip.x * frame_width)
        index_y = int(index_tip.y * frame_height)

        thumb_x = int(thumb_tip.x * frame_width)
        thumb_y = int(thumb_tip.y * frame_height)

        middle_x = int(middle_tip.x * frame_width)
        middle_y = int(middle_tip.y * frame_height)


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        cv2.circle(
            frame,
            (index_x, index_y),
            10,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            10,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (middle_x, middle_y),
            8,
            (0, 255, 255),
            -1
        )


        # ====================================================
        # DISTANCE: THUMB ↔ INDEX
        # ====================================================

        pinch_distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2
        )


        # ====================================================
        # DISTANCE: INDEX ↔ MIDDLE
        # ====================================================

        index_middle_distance = math.sqrt(
            (index_tip.x - middle_tip.x) ** 2 +
            (index_tip.y - middle_tip.y) ** 2
        )


        # ====================================================
        # FINGER STATES
        # ====================================================

        # Index is extended if fingertip is above its PIP joint
        index_extended = (
            index_tip.y < hand[6].y
        )

        # Middle finger extended
        middle_extended = (
            middle_tip.y < hand[10].y
        )

        # Ring finger folded
        ring_folded = (
            ring_tip.y > hand[14].y
        )

        # Pinky folded
        pinky_folded = (
            pinky_tip.y > hand[18].y
        )


        # ====================================================
        # TWO-FINGER GESTURE
        # ====================================================

        two_finger_gesture = (
            index_extended
            and middle_extended
            and ring_folded
            and pinky_folded
        )


        # ====================================================
        # PINCH
        # ====================================================

        pinch_gesture = (
            pinch_distance < PINCH_THRESHOLD
        )


        # ====================================================
        # RIGHT CLICK
        # ====================================================

        current_time = time.time()

        if two_finger_gesture:

            scrolling = True


            # -----------------------------------------------
            # RIGHT CLICK
            # -----------------------------------------------

            if (
                current_time - last_right_click_time
                > CLICK_COOLDOWN
            ):

                # Right click when two fingers first appear
                if not scrolling:

                    pyautogui.rightClick()

                    last_right_click_time = current_time


            # -----------------------------------------------
            # Scroll
            # -----------------------------------------------

            if previous_scroll_y is None:

                previous_scroll_y = (
                    index_y + middle_y
                ) / 2

            else:

                current_scroll_y = (
                    index_y + middle_y
                ) / 2

                movement = (
                    previous_scroll_y -
                    current_scroll_y
                )

                if abs(movement) > SCROLL_DEADZONE:

                    scroll_amount = int(
                        movement /
                        SCROLL_SENSITIVITY
                    )

                    if scroll_amount != 0:

                        pyautogui.scroll(
                            scroll_amount
                        )

                previous_scroll_y = current_scroll_y


            # Display gesture

            cv2.putText(
                frame,
                "SCROLL",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3
            )


        else:

            previous_scroll_y = None

            scrolling = False


        # ====================================================
        # NORMAL MOUSE MOVEMENT
        # ====================================================

        if not two_finger_gesture:

            # -----------------------------------------------
            # Convert camera → control area
            # -----------------------------------------------

            x = (
                index_x - FRAME_MARGIN_X
            ) / (
                frame_width -
                2 * FRAME_MARGIN_X
            )

            y = (
                index_y - FRAME_MARGIN_Y
            ) / (
                frame_height -
                2 * FRAME_MARGIN_Y
            )


            # Keep 0 → 1

            x = max(
                0.0,
                min(1.0, x)
            )

            y = max(
                0.0,
                min(1.0, y)
            )


            # -----------------------------------------------
            # Screen coordinates
            # -----------------------------------------------

            target_x = int(
                x * screen_width
            )

            target_y = int(
                y * screen_height
            )


            # -----------------------------------------------
            # Smooth movement
            # -----------------------------------------------

            mouse_x = int(
                previous_mouse_x +
                (
                    target_x -
                    previous_mouse_x
                ) * SMOOTHING
            )

            mouse_y = int(
                previous_mouse_y +
                (
                    target_y -
                    previous_mouse_y
                ) * SMOOTHING
            )


            # =================================================
            # DRAGGING
            # =================================================

            if pinch_gesture:

                # Start drag
                if not dragging:

                    pyautogui.mouseDown()

                    dragging = True


                # Move while holding
                pyautogui.moveTo(
                    mouse_x,
                    mouse_y,
                    _pause=False
                )


                cv2.putText(
                    frame,
                    "DRAGGING",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 165, 255),
                    3
                )


            else:

                # Release drag
                if dragging:

                    pyautogui.mouseUp()

                    dragging = False


                # =================================================
                # NORMAL MOVEMENT
                # =================================================

                pyautogui.moveTo(
                    mouse_x,
                    mouse_y,
                    _pause=False
                )


                # =================================================
                # LEFT CLICK
                # =================================================

                if not pinching:

                    if (
                        current_time -
                        last_click_time
                        > CLICK_COOLDOWN
                    ):

                        # A short pinch creates a click.
                        # Dragging is handled separately.

                        pass


                cv2.putText(
                    frame,
                    "MOVE",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    3
                )


            previous_mouse_x = mouse_x
            previous_mouse_y = mouse_y


        # ====================================================
        # LEFT CLICK / PINCH
        # ====================================================

        if pinch_gesture and not dragging:

            cv2.putText(
                frame,
                "PINCH",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ====================================================
        # DRAW PINCH LINE
        # ====================================================

        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (0, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY DISTANCES
        # ====================================================

        cv2.putText(
            frame,
            f"Pinch: {pinch_distance:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        # Release mouse if hand disappears
        if dragging:

            pyautogui.mouseUp()

            dragging = False


        previous_scroll_y = None

        cv2.putText(
            frame,
            "NO HAND",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Air Mouse",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

if dragging:

    pyautogui.mouseUp()


cap.release()

cv2.destroyAllWindows()

detector.close()

print("Air Mouse stopped.")