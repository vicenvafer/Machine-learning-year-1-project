import os
import cv2

# Set DATA_DIR to the "Pictures" directory in the user's home folder
DATA_DIR = #Directory path to choose

# Create the main directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 50

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))  # Create subdirectory for each class
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Apply the mirror flip horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Display the flipped image
        cv2.putText(flipped_frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', flipped_frame)

        # Break if 'Q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Display the flipped frame
        cv2.imshow('frame', flipped_frame)
        cv2.waitKey(25)

        # Save the flipped frame
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), flipped_frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
