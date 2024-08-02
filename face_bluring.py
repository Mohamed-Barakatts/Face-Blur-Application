import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Initialize the video capture object
cap = cv2.VideoCapture(0)
# Alternatively, you can specify a video file path to capture video from a file
# cap = cv2.VideoCapture("D:\openCV\videos\street6.mp4")

# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object #! DOWNLOADING IT LOCALY
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Create an instance of the FaceDetector class
fd = FaceDetector()

# Main loop to process each frame of the video
while True:
    # Read a frame from the video capture object
    ret, video = cap.read()
    
    # Detect faces in the frame
    img, faces = fd.findFaces(video)
    
    # If faces are detected, iterate through each face and blur it
    if faces:
        for face in faces:
            x, y, w, h = face["bbox"]
            # Modify y-coordinate to start from higher up on the forehead
            y -= int(h * 0.5)
            # Adjust height to cover the entire face from forehead to chin
            h += int(h * 0.5)
            # Make the bounding box wider
            x -= int(w * 0.2)
            w += int(w * 0.4)
            
            # Ensure the new coordinates are within the frame
            x = max(0, x)
            y = max(0, y)
            
            # Extract face region
            face_img = video[y:y+h, x:x+w]
            # Apply blur effect to the face region
            face_img = cv2.blur(face_img, (100, 100), 3)
            # Replace the original face region with the blurred face region
            video[y:y+h, x:x+w] = face_img
    
    # Write the processed frame to the output video file
    out.write(video)
    
    # Display the frame with the blurred faces
    cv2.imshow("frame", video)
    
    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture object, video writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()