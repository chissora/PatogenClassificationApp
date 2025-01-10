import gradio as gr
from tensorflow.keras.models import load_model
from ultralyticsplus import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import requests

# Ścieżki do modeli
model_path = "Classifier/PatoGenClassifier.keras"
yolo_model_path = 'foduucom/plant-leaf-detection-and-classification'

# Lista nazw roślin
plants_list = [
    'ginger', 'banana', 'tobacco', 'ornamaental', 'rose', 'soyabean', 'papaya',
    'garlic', 'raspberry', 'mango', 'cotton', 'corn', 'pomgernate', 'strawberry',
    'Blueberry', 'brinjal', 'potato', 'wheat', 'olive', 'rice', 'lemon', 'cabbage',
    'gauava', 'chilli', 'capcicum', 'sunflower', 'cherry', 'cassava', 'apple', 'tea',
    'sugarcane', 'groundnut', 'weed', 'peach', 'coffee', 'cauliflower', 'tomato',
    'onion', 'gram', 'chiku', 'jamun', 'castor', 'pea', 'cucumber', 'grape', 'cardamom'
]
plants_dict = {index: plant for index, plant in enumerate(plants_list)}

pathogen_classes = {
    0: "Bacterial Disease",
    1: "Fungal Infection",
    2: "Healthy Plant",
    3: "Viral Disease"
}

# Ładowanie modeli
classifier_model = load_model(model_path)
yolo_model = YOLO(yolo_model_path)

# Funkcja preprocessingu obrazów
def preprocess_image(cropped_image):
    resized_image = cropped_image.resize((224, 224))  # Rozmiar wejściowy zgodny z modelem
    array_image = img_to_array(resized_image) / 255.0  # Normalizacja pikseli
    input_image = np.expand_dims(array_image, axis=0)  # Dodanie wymiaru batch
    return input_image

# Główna funkcja analizy obrazu
def analyze_image(image):
    # Detekcja YOLO
    yolo_model.overrides['conf'] = 0.25
    yolo_model.overrides['iou'] = 0.45
    yolo_model.overrides['agnostic_nms'] = False
    yolo_model.overrides['max_det'] = 1000

    results = yolo_model.predict(image)
    boxes = results[0].boxes.data.numpy()  # [x_min, y_min, x_max, y_max, conf, class]

    detected_boxes = []
    pathogen_predictions = []

    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = map(int, box)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        input_image = preprocess_image(cropped_image)
        detected_boxes.append((input_image, cls))

    for input_image, yolo_class in detected_boxes:
        prediction = classifier_model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        pathogen_predictions.append((yolo_class, predicted_class, confidence))

    # Generowanie promptów
    prompts = []
    for yolo_class, predicted_class, confidence in pathogen_predictions:
        yolo_class_name = plants_dict.get(yolo_class, "Unknown Plant")
        pathogen_class_name = pathogen_classes.get(predicted_class, "Unknown Pathogen")
        if pathogen_class_name != "Healthy Plant":
            prompt = (
                f"A {pathogen_class_name} has been detected on a {yolo_class_name} "
                f"with a confidence of {confidence:.2f}. "
                f"How can I prevent or treat this disease?"
            )
        else:
            prompt = (
                f"The detected {yolo_class_name} is healthy with a confidence of {confidence:.2f}. "
                f"Do you have any suggestions for maintaining the health of this plant?"
            )
        prompts.append(prompt)

    return prompts[0] if prompts else "No plant or disease detected."

# Funkcja do wysyłania promptu do API
def send_to_api(prompt):
    # Przykładowa implementacja dla API OpenAI
    api_key = "YOUR_API_KEY"
    url = "https://api.openai.com/v1/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 100,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Konfiguracja Gradio
def interface_function(image):
    prompt = analyze_image(image)
    return prompt

def handle_accept(prompt):
    return send_to_api(prompt)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_button = gr.Button("Submit")
        with gr.Column():
            prompt_output = gr.Textbox(label="Generated Prompt", interactive=True)
            send_button = gr.Button("Accept and Send to API")
            api_response = gr.Textbox(label="API Response", interactive=False)
    
    # Obsługa zdarzeń
    submit_button.click(interface_function, inputs=input_image, outputs=prompt_output)
    send_button.click(handle_accept, inputs=prompt_output, outputs=api_response)

demo.launch()