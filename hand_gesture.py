import cv2
import mediapipe as mp
import pyautogui
from collections import deque
import time

# ------------------- Setup -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

# Cursor smoothing
pts_x = deque(maxlen=5)
pts_y = deque(maxlen=5)

# Scroll control
prev_middle_y = None
scroll_delay = 0.08
last_scroll_time = 0
scroll_threshold = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape

        # Landmarks
        thumb = hand.landmark[4]
        index = hand.landmark[8]
        middle = hand.landmark[12]
        index_mcp = hand.landmark[5]
        middle_mcp = hand.landmark[9]

        # ---------------- CURSOR (Index finger) ----------------
        if index.y < index_mcp.y:
            sx = int(index.x * screen_w)
            sy = int(index.y * screen_h)
            pts_x.append(sx)
            pts_y.append(sy)
            pyautogui.moveTo(sum(pts_x)//len(pts_x), sum(pts_y)//len(pts_y))
            cv2.circle(frame, (int(index.x*w), int(index.y*h)), 10, (255,0,0), -1)

        # ---------------- CLICK (Pinch) ----------------
        pinch_dist = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
        if pinch_dist < 0.045:
            pyautogui.click()
            time.sleep(0.25)

        # ---------------- SCROLL (Middle finger only) ----------------
        if middle.y < middle_mcp.y and index.y > index_mcp.y:
            current_time = time.time()
            middle_y = int(middle.y * screen_h)

            if prev_middle_y is None:
                prev_middle_y = middle_y

            if current_time - last_scroll_time > scroll_delay:
                diff = prev_middle_y - middle_y
                if abs(diff) > scroll_threshold:
                    pyautogui.scroll(30 if diff > 0 else -30)
                    prev_middle_y = middle_y
                    last_scroll_time = current_time

            cv2.putText(frame, "SCROLL MODE", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            prev_middle_y = None

    cv2.imshow("Air Touch (Stable)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
