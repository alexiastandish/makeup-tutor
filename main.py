

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import openai
import os
import numpy as np
from dotenv import load_dotenv

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')



load_dotenv()

# Initialize OpenAI API

openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

# Function to extract facial features using Mediapipe
def get_facial_features(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded, please check the image path.")
    
    # Convert to RGB (as Mediapipe expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    facial_features = {}

    if results.multi_face_landmarks:
        print(results)
        # Extract facial landmarks
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            facial_features[face_id] = landmarks

    return facial_features

def classify_face_shape(facial_features):

    face_id = list(facial_features.keys())[0]
    facial_landmarks = facial_features[face_id]

    # Key facial distances
    forehead_width = np.linalg.norm(np.array(facial_landmarks[10]) - np.array(facial_landmarks[338]))  # Forehead width
    cheekbone_width = np.linalg.norm(np.array(facial_landmarks[234]) - np.array(facial_landmarks[454]))  # Cheekbone width
    jawline_width = np.linalg.norm(np.array(facial_landmarks[172]) - np.array(facial_landmarks[397]))  # Jaw width
    face_length = np.linalg.norm(np.array(facial_landmarks[10]) - np.array(facial_landmarks[152]))  # Face length

    # Determine face shape
    if face_length > cheekbone_width * 1.5:
        return "oblong"
    elif abs(cheekbone_width - jawline_width) < 10 and abs(cheekbone_width - forehead_width) < 10:
        return "square"
    elif cheekbone_width > forehead_width and cheekbone_width > jawline_width:
        return "diamond"
    elif forehead_width > cheekbone_width > jawline_width:
        return "heart"
    elif cheekbone_width > jawline_width:
        return "oval"
    else:
        return "round"



def classify_eye_shape(facial_features):
    """
    Classifies the eye shape based on facial landmarks from MediaPipe.
    
    Eye shapes include: almond, round, hooded, monolid, upturned, and downturned.
    """
    face_id = list(facial_features.keys())[0]
    landmarks = facial_features[face_id]

    # Left eye landmarks
    left_eye_outer = np.array(landmarks[33])   # Outer corner
    left_eye_inner = np.array(landmarks[133])  # Inner corner
    left_eye_top = np.array(landmarks[159])    # Upper eyelid
    left_eye_bottom = np.array(landmarks[145]) # Lower eyelid

    # Right eye landmarks
    right_eye_outer = np.array(landmarks[362])  
    right_eye_inner = np.array(landmarks[263])  
    right_eye_top = np.array(landmarks[386])    
    right_eye_bottom = np.array(landmarks[374]) 

    # Eye width and height
    left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
    left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)

    right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
    right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)

    # Average eye width-to-height ratio
    eye_ratio = ((left_eye_width / left_eye_height) + (right_eye_width / right_eye_height)) / 2

    # Eyelid curvature (difference between top and bottom lid heights)
    left_curvature = left_eye_top[1] - left_eye_bottom[1]
    right_curvature = right_eye_top[1] - right_eye_bottom[1]
    curvature_avg = (left_curvature + right_curvature) / 2

    # Outer corner position (Upturned/Downturned)
    left_corner_tilt = left_eye_outer[1] - left_eye_inner[1]
    right_corner_tilt = right_eye_outer[1] - right_eye_inner[1]
    tilt_avg = (left_corner_tilt + right_corner_tilt) / 2

    # Classifying eye shape
    if eye_ratio >= 2.5:
        return "almond"
    elif eye_ratio < 1.8:
        return "round"
    elif curvature_avg < 0:  # Negative curvature = upper eyelid covering more
        return "hooded"
    elif eye_ratio > 2.0 and curvature_avg >= 0:
        return "monolid"
    elif tilt_avg < -2:  # Outer corners lower than inner corners
        return "downturned"
    elif tilt_avg > 2:  # Outer corners higher than inner corners
        return "upturned"
    else:
        return "almond"


def classify_lip_shape(facial_features):
    """
    Classifies the lip shape based on facial landmarks from MediaPipe.
    
    Lip shapes include: full, thin, heart-shaped, wide, downturned, and upturned.
    """
    face_id = list(facial_features.keys())[0]
    landmarks = facial_features[face_id]

    # Key lip landmarks
    mouth_left = np.array(landmarks[61])   # Left corner of mouth
    mouth_right = np.array(landmarks[291]) # Right corner of mouth
    upper_lip_top = np.array(landmarks[13])  # Top center of upper lip (cupid's bow)
    lower_lip_bottom = np.array(landmarks[14])  # Bottom center of lower lip
    upper_lip_mid = np.array(landmarks[0])   # Midpoint of upper lip (philtrum area)

    # Lip width and height
    lip_width = np.linalg.norm(mouth_right - mouth_left)
    lip_height = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    # Lip width-to-height ratio
    lip_ratio = lip_width / lip_height

    # Cupid's bow curvature (philtrum depth)
    cupid_bow_curvature = upper_lip_mid[1] - upper_lip_top[1]

    # Mouth corner tilt (Upturned/Downturned)
    mouth_corner_tilt = mouth_left[1] - mouth_right[1]

    # Classifying lip shape
    if lip_ratio < 1.5:
        return "full"  # Lips are more vertically prominent
    elif lip_ratio > 3.0:
        return "wide"  # Lips are much wider than tall
    elif cupid_bow_curvature > 2.5:
        return "heart-shaped"  # Well-defined cupid’s bow
    elif lip_height < 10:
        return "thin"  # Very small vertical height
    elif mouth_corner_tilt < -2:  # Left side lower than right side
        return "downturned"
    elif mouth_corner_tilt > 2:  # Right side lower than left side
        return "upturned"
    else:
        return "balanced"

def classify_eye_color(processed_img, facial_features):
    """
    Classifies eye color based on the dominant color in the iris region.

    Eye colors include: brown, blue, green, hazel, gray, amber.

    Parameters:
        processed_img (numpy array): The input image containing the face.
        facial_features (dict): Detected facial landmarks from MediaPipe.

    Returns:
        str: Classified eye color.
    """
    face_id = list(facial_features.keys())[0]
    landmarks = facial_features[face_id]

    # Ensure there are enough landmarks detected
    if len(landmarks) < 468:
        return "Insufficient landmarks for eye color classification"

    # Define the eye landmarks range
    left_eye_landmarks = [landmarks[i] for i in range(33, 134)]  # Approximate left eye landmarks
    right_eye_landmarks = [landmarks[i] for i in range(362, 464)]  # Approximate right eye landmarks

    # Calculate the centroid of the left and right eyes to approximate the iris center
    left_iris_center = np.mean(left_eye_landmarks, axis=0)
    right_iris_center = np.mean(right_eye_landmarks, axis=0)

    # Extract pixel colors from the processed_img at the iris locations
    left_eye_color = processed_img[int(left_iris_center[1]), int(left_iris_center[0])]
    right_eye_color = processed_img[int(right_iris_center[1]), int(right_iris_center[0])]

    # Average the two eye colors
    avg_eye_color = np.mean([left_eye_color, right_eye_color], axis=0)

    # Convert BGR to HSV for better color classification
    avg_eye_color_hsv = cv2.cvtColor(np.uint8([[avg_eye_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Classify based on HSV values
    h, s, v = avg_eye_color_hsv
    if v < 50:
        return "dark brown"
    elif h < 25 and s > 50:
        return "amber"
    elif h < 30:
        return "brown"
    elif 30 <= h <= 85:
        return "green"
    elif 85 < h <= 130:
        return "blue"
    elif 130 < h <= 160:
        return "gray"
    else:
        return "hazel"

# Function to generate makeup tutorial based on facial features
def generate_makeup_tutorial(facial_features, processed_img):
    print('facial_features', facial_features)
    # Simplified approach: Use key features for the makeup tutorial
    face_shape = classify_face_shape(facial_features)
    print('face_shape', face_shape)
    eye_shape = classify_eye_shape(facial_features)  
    print('eye_shape', eye_shape)
    lip_shape = classify_lip_shape(facial_features) 
    print('lip_shape', lip_shape)
    eye_color = classify_eye_color(processed_img, facial_features)
    
    # Simplify the prompt
    user_prompt = f"""
    Create a personalized makeup tutorial based on these features:
    - Face Shape: {face_shape}
    - Eye Shape: {eye_shape}
    - Lip Shape: {lip_shape}
    
    Provide a step-by-step guide for makeup, including foundation, contouring, eye makeup, and lips.
    """
    
    # Prepare the messages for the chat endpoint
    messages = [
        {"role": "system", "content": "You are a professional makeup artist."},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call the Chat Completion API using the correct endpoint
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo" if preferred
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    
    # Extract the generated tutorial from the response
    tutorial = response['choices'][0]['message']['content'].strip()
    # return tutorial
    return {
        "tutorial": tutorial,
        "features": {
            "eye_color": eye_color,
            "lip_shape": lip_shape,
            "face_shape": face_shape,
            "eye_shape": eye_shape,
        }
    }


@app.route('/get-tutorial', methods=['POST'])
def get_tutorial():
    print('hihihihih')
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image = request.files['image']
    print('image imageimage', image)
    print(f"Image size: {len(image.read())} bytes")
    image.seek(0)  
    # Save the image to a file
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)
    print(f"Saved image at: {image_path}")

    # Read the image data again
    image.seek(0)  # Reset the pointer to the start before reading
    image_bytes = image.read()
    print(f"Length of image bytes: {len(image_bytes)}")  # Check if the file was read correctly

    if not image_bytes:
        return jsonify({'error': 'Empty image file'}), 400  # Check if the file is empty

    # Convert to a NumPy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    processed_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if processed_img is None:
        return jsonify({'error': 'Failed to decode image'}), 400  # Check if decoding fails

    # Image successfully decoded, now process it
    print('Image processed by cv2:', processed_img)

    # Extract facial features from the uploaded image
    facial_features = get_facial_features(image_path)
    print(facial_features)

    # Generate the makeup tutorial based on facial features
    result = generate_makeup_tutorial(facial_features, processed_img)
    print(result)
    # Clean up the saved image after processing
    os.remove(image_path)

    if result:
        return jsonify(result), 200
    else:
        return jsonify({'error': 'Failed to generate tutorial'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
