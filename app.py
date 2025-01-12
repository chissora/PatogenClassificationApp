import gradio as gr
from tensorflow.keras.models import load_model
from ultralyticsplus import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import openai
from openai import OpenAI

# Paths to the models
model_path = "Classifier/PatoGenClassifier.keras"
yolo_model_path = 'foduucom/plant-leaf-detection-and-classification'

# List of plant names
plants_list = [
    'ginger', 'banana', 'tobacco', 'ornamaental', 'rose', 'soyabean', 'papaya',
    'garlic', 'raspberry', 'mango', 'cotton', 'corn', 'pomgernate', 'strawberry',
    'Blueberry', 'brinjal', 'potato', 'wheat', 'olive', 'rice', 'lemon', 'cabbage',
    'gauava', 'chilli', 'capcicum', 'sunflower', 'cherry', 'cassava', 'apple', 'tea',
    'sugarcane', 'groundnut', 'weed', 'peach', 'coffee', 'cauliflower', 'tomato',
    'onion', 'gram', 'chiku', 'jamun', 'castor', 'pea', 'cucumber', 'grape', 'cardamom'
]
plants_dict = {index: plant for index, plant in enumerate(plants_list)}

# Pathogen classes
pathogen_classes = {
    0: "Bacterial Disease",
    1: "Fungal Infection",
    2: "Healthy Plant",
    3: "Viral Disease"
}

# Load models
classifier_model = load_model(model_path)
yolo_model = YOLO(yolo_model_path)

# Image preprocessing function
def preprocess_image(cropped_image):
    resized_image = cropped_image.resize((224, 224))
    array_image = img_to_array(resized_image) / 255.0
    input_image = np.expand_dims(array_image, axis=0)
    return input_image

# Main image analysis function
def analyze_image(image):
    yolo_model.overrides['conf'] = 0.25
    yolo_model.overrides['iou'] = 0.45

    results = yolo_model.predict(image)
    boxes = results[0].boxes.data.numpy()

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

# Load API key from file
key_file_path = os.path.join(".env", "OPENAI_API_KEY.txt")
with open(key_file_path, "r") as key_file:
    APIKey = key_file.read().strip()

# Ensure the API key is loaded correctly
if not APIKey:
    raise ValueError("The API key file is empty or invalid.")

# Initialize OpenAI client
client = OpenAI(api_key=APIKey)

# Define system prompt
system_prompt = "I am a plant care expert. I answer questions about plant diseases and care."

# Function to send prompt to API
def send_to_api(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the model you're using
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        # Extract the content from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio configuration
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
            # Smaller label using Markdown and styled with CSS
            gr.Markdown("**API Response from OpenAI**", elem_id="small-label")
            api_response = gr.Markdown(elem_id="scrollable-box")
    
    # Event handling
    submit_button.click(interface_function, inputs=input_image, outputs=prompt_output)
    send_button.click(handle_accept, inputs=prompt_output, outputs=api_response)

    # Add CSS for the smaller label and scrollable Markdown box
    demo.css = """
    #small-label {
        font-size: 14px;
        font-weight: bold;
        text-align: left;
        margin-bottom: 5px;
        color: #333;
    }
    #scrollable-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.1);
        font-size: 14px;
    }
    """
demo.launch()