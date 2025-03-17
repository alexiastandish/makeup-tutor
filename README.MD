## Makeup Tutorial Generator API

### Overview

The **Makeup Tutorial Generator API** is a Flask-based service that processes user-uploaded images, detects facial features using MediaPipe, and generates personalized makeup tutorials using OpenAI's GPT models.

## Features

- Detects **face shape, eye shape, lip shape, and eye color** from an uploaded image.
- Uses **OpenAI's GPT** to generate a step-by-step makeup tutorial tailored to the user's features.
- Supports **CORS** for integration with frontend applications.
- Provides an easy-to-use **REST API endpoint**.

## Dependencies

- Python 3.x
- Flask
- Flask-CORS
- OpenCV (`cv2`)
- MediaPipe
- OpenAI API key
- NumPy
- dotenv

### API Endpoint

`POST /get-tutorial`

- Uploads an image, analyzes facial features, and returns a personalized makeup tutorial.

#### Request:

- Form-data with an image file (key: image).

#### Response

```{
  "tutorial": "Step-by-step makeup guide...",
  "features": {
    "eye_color": "blue",
    "lip_shape": "full",
    "face_shape": "oval",
    "eye_shape": "almond"
  }
}
```

## Notes

- The /uploads directory is created automatically for storing temporary images.
- Processed images are deleted after the tutorial is generated.
- The API uses OpenAI's GPT-4 (or GPT-3.5) for tutorial generation.
