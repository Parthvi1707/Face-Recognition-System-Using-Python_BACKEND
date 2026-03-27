# Face Recognition System - Backend API

A robust, modern Python backend designed to detect human faces within images. It is built using **FastAPI** for lightning-fast asynchronous requests, and **OpenCV** (Haar Cascades) for high accuracy facial detection. 

This repository serves as the official integration endpoint for frontend web and mobile applications looking to implement face detection natively.

---

## 🚀 Features

- **Fast & Async**: Powered by FastAPI, ensuring high-frequency concurrent traffic handling.
- **CORS Enabled**: Set up explicitly to allow universal frontend interaction (React, Vue, Vite, etc.) without typical browser port blocking.
- **Two Modalities**:
  - `JSON Coordinates`: Returns precise mathematical `x, y, width, height` boundaries for every face found.
  - `Pre-Rendered Image`: Alternatively returns a raw `.jpg` with bright green rectangles already drawn on the faces, bypassing heavy client-side Canvas operations.
- **Built-in Docs**: Automatic Swagger UI testing environment generated out of the box.

---

## 🛠️ Installation & Setup

You will need Python 3 installed on your machine. We strongly recommend setting up a virtual environment to prevent package conflicts.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Parthvi1707/Face-Recognition-System-Using-Python_BACKEND.git
   cd Face-Recognition-System-Using-Python_BACKEND
   ```

2. **Create a Virtual Environment:**
   *(For Windows)*
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   *(For Mac/Linux)*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Running the API

Once the environment is set up and activated, boot the development server with:

```bash
uvicorn main:app --reload
```

The server will spin up actively at `http://localhost:8000`. 
Head to **`http://localhost:8000/docs`** in your browser to interactively test all endpoints using the built-in Swagger UI!

---

## 📡 Endpoints Overview

| Method | Endpoint | Description | Response Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Checks if the API is active. | JSON |
| `POST` | `/api/detect-faces` | Accepts an image upload. Returns the bounding box coordinates of all faces detected. | JSON `[{"x":10, "y":10, ...}]` |
| `POST` | `/api/detect-image` | Accepts an image upload. Draws green rectangles around the faces and serves the modified image back. | Direct `image/jpeg` File |

> **Frontend Request Details:** 
> When hitting `POST` endpoints, ensure your frontend client (like `fetch` or `axios`) uploads the binary image by appending it to a `FormData` object under the key `"file"`.
