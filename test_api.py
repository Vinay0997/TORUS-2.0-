import requests

BASE_URL = "http://127.0.0.1:8000"

sample_exam = {
    "age": 52,
    "sex": "F",
    "risk_category": "high",
    "device_id": "DEV-003",
    "exam_type": "cardiac",
    "image_quality_score": 0.78
}

response = requests.post(f"{BASE_URL}/predict", json=sample_exam)
print("Status code:", response.status_code)
print("Response JSON:", response.json())
