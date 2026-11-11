import cv2
import mediapipe as mp
import pygame
import time
import math

# ================= AUDIO =================
pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)

notes = {
    "C":  "sounds/violin_C.mp3",
    "D":  "sounds/violin_D.mp3",
    "E":  "sounds/violin_E.mp3",
    "F":  "sounds/violin_F.mp3",
    "G":  "sounds/violin_G.mp3",
    "A":  "sounds/violin_A.mp3",
    "B":  "sounds/violin_B.mp3",
    "C5": "sounds/violin_C5.mp3"
}

current_note = None
prev_bow_x = None
sound_playing = False

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= NOTE ZONES =================
def get_note(y, h):
    zones = [
        ("C5", 0.12), ("B", 0.22), ("A", 0.32), ("G", 0.45),
        ("F", 0.58), ("E", 0.70), ("D", 0.82), ("C", 1.00)
    ]
    for note, limit in zones:
        if y < h * limit:
            return note

# ================= DRAW 3D VIOLIN BODY =================
def draw_3d_violin_body(frame, cx, cy):
    # back shadow
    cv2.ellipse(frame, (cx+6, cy+10), (90, 140), 0, 0, 360, (40, 40, 40), -1)

    # main body
    cv2.ellipse(frame, (cx, cy), (90, 140), 0, 0, 360, (80, 40, 20), -1)

    # highlight
    cv2.ellipse(frame, (cx-25, cy-40), (20, 40), 0, 0, 360, (140, 80, 50), -1)

    # fingerboard
    cv2.rectangle(frame, (cx-18, cy-170), (cx+18, cy-40), (30, 30, 30), -1)

# ================= DRAW 3D STRINGS =================
def draw_3d_strings(frame, cx, cy, active_note, bowing):
    strings = ["G", "D", "A", "E"]
    depth = [0, 2, 4, 6]  # depth illusion
    spacing = 14

    for i, s in enumerate(strings):
        y = cy - 160 + i * spacing
        z = depth[i]

        vib = int(2 * math.sin(time.time()*35)) if bowing and active_note.startswith(s) else 0

        color = (0, 255, 255) if active_note.startswith(s) else (220, 220, 220)
        thickness = 3 if active_note.startswith(s) else 2

        cv2.line(
            frame,
            (cx - 12 + z, y + vib),
            (cx + 12 + z, y - vib),
            color,
            thickness
        )

# ================= PLAY NOTE =================
def play_note(note):
    pygame.mixer.music.stop()
    pygame.mixer.music.load(notes[note])
    pygame.mixer.music.play(-1)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        string_hand = result.multi_hand_landmarks[0]
        bow_hand = result.multi_hand_landmarks[1]

        mp_draw.draw_landmarks(frame, string_hand, mp_hands.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, bow_hand, mp_hands.HAND_CONNECTIONS)

        finger = string_hand.landmark[8]
        fx, fy = int(finger.x * w), int(finger.y * h)
        note = get_note(fy, h)

        bow_x = int(bow_hand.landmark[8].x * w)
        bowing = False

        if prev_bow_x is not None and abs(bow_x - prev_bow_x) > 15:
            bowing = True
            if note != current_note:
                play_note(note)
                current_note = note
                sound_playing = True

        prev_bow_x = bow_x

        # Draw 3D violin
        draw_3d_violin_body(frame, fx, fy)
        draw_3d_strings(frame, fx, fy, note, bowing)

        cv2.putText(frame, f"Playing: {note}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    else:
        if sound_playing:
            pygame.mixer.music.stop()
            sound_playing = False
            current_note = None
        prev_bow_x = None

    cv2.putText(frame,
                "3D Virtual Violin | Bow ↔ | Strings ↑↓ | ESC Exit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("3D Virtual Violin", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

