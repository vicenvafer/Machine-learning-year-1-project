import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('#Directory path to choose, 'rb'))
model = model_dict['model']

# Initialize camera and mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.6)

# Label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    flipped_rgb = cv2.flip(frame_rgb, 1)
    flipped_frame = cv2.flip(frame, 1)

    results = hands.process(flipped_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        for hand_landmarks_draw in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                flipped_frame,
                hand_landmarks_draw,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Get prediction and probabilities
        probs = model.predict_proba([np.asarray(data_aux)])[0]
        predicted_index = np.argmax(probs)
        predicted_character = labels_dict[predicted_index]

        # Draw rectangle and predicted character
        cv2.putText(flipped_frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)

        # Display top 5 predictions
        top_n = 5
        top_indices = np.argsort(probs)[::-1][:top_n]

        for i, idx in enumerate(top_indices):
            label = labels_dict[idx]
            prob = probs[idx]
            cv2.putText(flipped_frame, f"{label}: {prob*100:.2f}%", (10, 30 + (i+1)*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (101, 101, 101), 1, cv2.LINE_AA)

    cv2.imshow('frame', flipped_frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
