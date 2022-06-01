import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
# import RPi.GPIO as GPIO

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands function.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, display = True):
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)
    
    # Check if landmarks are found.
    if results.multi_hand_landmarks:
        
        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                  connections = mp_hands.HAND_CONNECTIONS) 
    
    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image and results of hands landmarks detection.
        return output_image, results


def countFingers(image, results, draw=True, display=True):
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        
        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                
                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1
        
        # Retrieve the x-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:
 
        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20,255,155), 10, 10)
 
    # Check if the output image is specified to be displayed.
    if display:
        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1])
        plt.title("Output Image")
        plt.axis('off')
    # Otherwise
    else:
        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image, fingers_statuses, count



# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(1) #the order number of camera
camera_video.set(3,1280) #set resolution
camera_video.set(4,960)
# Create named window for resizing purposes.
cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)
 
# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    # Read a frame.
    ok, frame = camera_video.read()
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)
    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, display=False)
                
    # Display the frame.
    cv2.imshow('Fingers Counter', frame)
    # Check if 'q' is pressed and break the loop.
    if cv2.waitKey(1) == ord('q'):
        break
 
# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BOARD)

# Create the pin output for 7 segments led
# GPIO.setup(40, GPIO.OUT)
# GPIO.setup(37, GPIO.OUT)
# GPIO.setup(35, GPIO.OUT)
# GPIO.setup(33, GPIO.OUT)
# GPIO.setup(31, GPIO.OUT)
# GPIO.setup(29, GPIO.OUT)
# GPIO.setup(23, GPIO.OUT)

# digitclr=[0,0,0,0,0,0,0]
# digit0=[1,1,1,1,1,1,0]
# digit1=[0,1,1,0,0,0,0]
# digit2=[1,1,0,1,1,0,1]
# digit3=[1,1,1,1,0,0,1]
# digit4=[0,1,1,0,0,1,1]
# digit5=[1,0,1,1,0,1,1]
# digit6=[1,0,1,1,1,1,1]
# digit7=[1,1,1,0,0,0,0]
# digit8=[1,1,1,1,1,1,1]
# digit9=[1,1,1,0,0,1,1]

# gpin=[40,37,35,33,31,29,23]

# the loop for 7 segments clear and display
# def digdisp(digit):
#     for x in range(0,7):
#         GPIO.output(gpin[x], digitclr[x])
#     for x in range(0,7):
#         GPIO.output(gpin[x], digit[x])

# if(count == 0):
#     digdisp(digit0)
# if(count == 1):
#     digdisp(digit1)
# if(count == 2):
#     digdisp(digit2)
# if(count == 3):
#     digdisp(digit3)
# if(count == 4):
#     digdisp(digit4)
# if(count == 5):
#     digdisp(digit5)
# if(count == 6):
#     digdisp(digit6)
# if(count == 7):
#     digdisp(digit7)
# if(count == 8):
#     digdisp(digit8)
# if(count == 9):
#     digdisp(digit9)
# GPIO.cleanup()

