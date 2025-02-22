
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import mediapipe as mp
# import openai
# import sys
# import json
# import numpy as np
# import os


# # Set your OpenAI API Key
# openai.api_key = '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'


# # Function to extract facial features using Mediapipe
# def get_facial_features(image_path):
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh()

#     # Read the image using OpenCV
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image could not be loaded, please check the image path.")
    
#     # Convert to RGB (as Mediapipe expects RGB images)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(image_rgb)

#     facial_features = {}

#     if results.multi_face_landmarks:
#         # Extract facial landmarks
#         for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
#             landmarks = []
#             for landmark in face_landmarks.landmark:
#                 landmarks.append([landmark.x, landmark.y, landmark.z])
#             facial_features[face_id] = landmarks

#     return facial_features

# # Function to classify the face shape based on extracted features
# def classify_face_shape(facial_features):
#     # Simplified face shape classification using the distance between key points
#     face_id = list(facial_features.keys())[0]
#     facial_landmarks = facial_features[face_id]
    
#     cheekbone_distance = np.linalg.norm(np.array(facial_landmarks[234]) - np.array(facial_landmarks[454]))
#     jawline_distance = np.linalg.norm(np.array(facial_landmarks[0]) - np.array(facial_landmarks[17]))
    
#     if cheekbone_distance > jawline_distance:
#         return "oval"
#     else:
#         return "round"


# def generate_makeup_tutorial(facial_features):
#     # Simplified approach: Use key features for the makeup tutorial
#     face_shape = classify_face_shape(facial_features)
#     eye_shape = facial_features.get("eye_shape", "round")  # Default if not found
#     lip_shape = facial_features.get("lip_shape", "full")  # Default if not found
    
#     # Simplify the prompt
#     user_prompt = f"""
#     Create a personalized makeup tutorial based on these features:
#     - Face Shape: {face_shape}
#     - Eye Shape: {eye_shape}
#     - Lip Shape: {lip_shape}
    
#     Provide a step-by-step guide for makeup, including foundation, contouring, eye makeup, and lips.
#     """
    
#     # Prepare the messages for the chat endpoint
#     messages = [
#         {"role": "system", "content": "You are a professional makeup artist."},
#         {"role": "user", "content": user_prompt}
#     ]
    
#     # Call the Chat Completion API using the correct endpoint
#     response = openai.ChatCompletion.create(
#         model="gpt-4",  # or "gpt-3.5-turbo" if preferred
#         messages=messages,
#         max_tokens=1000,
#         temperature=0.7
#     )
    
#     # Extract the generated tutorial from the response
#     tutorial = response['choices'][0]['message']['content'].strip()
#     return tutorial


# # Main function to process the image and generate tutorial
# def main(image_path):
#     try:
#         # Step 1: Get facial features from the image
#         facial_features = get_facial_features(image_path)
        
#         # Step 2: Generate the makeup tutorial based on facial features
#         tutorial = generate_makeup_tutorial(facial_features)
        
#         # Step 3: Display the generated tutorial in the terminal
#         print("Your Makeup Tutorial:")
#         print(tutorial)
    
#     except Exception as e:
#         print(f"Error: {e}")

# # if __name__ == "__main__":
# #     if len(sys.argv) < 2:
# #         print("Usage: python script.py <image_path>")
# #     else:
# #         image_path = sys.argv[1]
# #         main(image_path)


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import openai
import os
import numpy as np


# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')




# Initialize OpenAI API
openai.api_key = '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'

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
        # Extract facial landmarks
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            facial_features[face_id] = landmarks

    return facial_features

def classify_face_shape(facial_features):
    # Simplified face shape classification using the distance between key points
    face_id = list(facial_features.keys())[0]
    facial_landmarks = facial_features[face_id]
    
    cheekbone_distance = np.linalg.norm(np.array(facial_landmarks[234]) - np.array(facial_landmarks[454]))
    jawline_distance = np.linalg.norm(np.array(facial_landmarks[0]) - np.array(facial_landmarks[17]))
    
    if cheekbone_distance > jawline_distance:
        return "oval"
    else:
        return "round"


# Function to generate makeup tutorial based on facial features
def generate_makeup_tutorial(facial_features):
    # Simplified approach: Use key features for the makeup tutorial
    face_shape = classify_face_shape(facial_features)
    eye_shape = facial_features.get("eye_shape", "round")  # Default if not found
    lip_shape = facial_features.get("lip_shape", "full")  # Default if not found
    
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
    return tutorial

# Endpoint to handle image upload and generate makeup tutorial
@app.route('/get-tutorial', methods=['POST'])

def get_tutorial():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    print(request.files)
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)
    
    # Extract facial features from the uploaded image
    facial_features = get_facial_features(image_path)

    # Generate the makeup tutorial based on facial features
    tutorial = generate_makeup_tutorial(facial_features)
    
    # Clean up the saved image after processing
    os.remove(image_path)
    
    if tutorial:
        return jsonify({'tutorial': tutorial}), 200
    else:
        return jsonify({'error': 'Failed to generate tutorial'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
