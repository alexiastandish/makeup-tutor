import cv2
import mediapipe as mp
import numpy as np

def get_facial_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize Mediapipe face detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    # Process the image and get landmarks
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        raise Exception("No faces detected in the image")
    
    # Assuming single face is detected
    landmarks = results.multi_face_landmarks[0]
    
    # Convert landmarks to a more accessible form (e.g., list of (x, y) coordinates)
    facial_features = []
    for landmark in landmarks.landmark:
        facial_features.append((landmark.x, landmark.y))
    
    return facial_features

def classify_face_shape(facial_features):
    # Simplified classification based on distances between key points (e.g., cheekbones, jawline, etc.)
    # This can be enhanced to classify more accurately by comparing proportions of face features
    cheekbone_distance = np.linalg.norm(np.array(facial_features[234]) - np.array(facial_features[454]))  # Example points
    jawline_distance = np.linalg.norm(np.array(facial_features[0]) - np.array(facial_features[17]))  # Example points

    if cheekbone_distance > jawline_distance:
        return "oval"
    else:
        return "round"



# import cv2
# import mediapipe as mp

# # Initialize Mediapipe Face Mesh model
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()

# def get_facial_features(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Process the image to get facial landmarks
#     results = face_mesh.process(image_rgb)

#     # Extract facial features from landmarks
#     facial_features = {}

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Example: Calculate the distance between two eyes
#             left_eye = face_landmarks.landmark[33]  # Left eye point (for example)
#             right_eye = face_landmarks.landmark[133]  # Right eye point (for example)

#             eye_distance = ((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2) ** 0.5

#             # Store facial features
#             facial_features["eye_distance"] = eye_distance
#             # Add more features like nose, lips, etc., based on the landmarks you need.

#     return facial_features


# # import cv2
# # import mediapipe as mp

# # # Initialize Mediapipe Face Mesh
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# # mp_drawing = mp.solutions.drawing_utils

# # # Load an image using OpenCV (replace with your image file path)
# # image = cv2.imread("your_image.jpg")  # Change this to your image path

# # # Check if image is loaded successfully
# # if image is None:
# #     print("Error: Image not found or could not be loaded.")
# #     exit()

# # # Convert the image to RGB as OpenCV uses BGR by default
# # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # # Process the image to detect facial landmarks
# # results = face_mesh.process(image_rgb)

# # # Draw the facial landmarks on the image
# # if results.multi_face_landmarks:
# #     for face_landmarks in results.multi_face_landmarks:
# #         mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

# # # Display the image with landmarks
# # cv2.imshow("Facial Landmarks", image)

# # # Wait for a key press and close the window
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # # Clean up
# # face_mesh.close()
