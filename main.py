# FastAPI Server: main.py
from fastapi import FastAPI, Request
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(angle, -1.0, 1.0)) / np.pi * 180
    return angle


def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 1 and up_fingers[0] == 8:
        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]
        angle = get_angle(v1, v2)
        if angle < 160 
            return "1"
    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        return "6" //good
    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        return "8" //bad
    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        return "9" //fuck
    elif len(up_fingers) == 2 and up_fingers == [8, 12]:
        return "2"
    elif len(up_fingers) == 3 and up_fingers == [8, 12, 16]:
        return "3"
    elif len(up_fingers) == 3 and up_fingers == [4, 8, 12]:
        dis_8_12 = np.linalg.norm(list_lms[8] - list_lms[12])
        dis_4_12 = np.linalg.norm(list_lms[4] - list_lms[12])
        ratio = dis_4_12 / (dis_8_12 + 1)
        return "7" //gun
    elif len(up_fingers) == 3 and up_fingers == [4, 8, 20]:
        return "10" //rock
    elif len(up_fingers) == 4 and up_fingers == [8, 12, 16, 20]:
        return "4"
    elif len(up_fingers) == 5:
        return "5"
    return " "


def process_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Invalid image"}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return {"status": "no_hand_detected"}

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    up_fingers = []
    finger_tips = [4, 8, 12, 16, 20]
    for i in finger_tips:
        if i == 4:
            if landmarks[4][0] > landmarks[3][0]:
                up_fingers.append(i)
        else:
            if landmarks[i][1] < landmarks[i - 2][1]:
                up_fingers.append(i)

    gesture = get_str_guester(up_fingers, landmarks)
    return {"status": "success", "gesture": gesture}


@app.post("/upload/")
async def upload(request: Request):
    img_bytes = await request.body()
    result = process_image(img_bytes)
    return result




