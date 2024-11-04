import gradio as gr
import skops.io as sio
import os

print(os.getcwd())

# Load the model
model_path = "Model/drug_pipeline.skops"
try:

    trusted_types = sio.get_untrusted_types()
    pipe = sio.load(model_path, trusted=trusted_types)

    if not hasattr(pipe, 'predict'):
        raise AttributeError("The loaded model does not have a 'predict' method.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

# Your training code here
def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.

    Args:
        age (int): Age of patient
        sex (str): Sex of patient
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood

    Returns:
        str: Predicted drug label
    """
    if pipe is None:
        return "Model not loaded correctly."

    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    try:
        predicted_drug = pipe.predict([features])[0]
        print(predicted_drug)
        label = f"Predicted Drug: {predicted_drug}"
        return label
    except Exception as e:
        return f"Error during prediction: {e}"

inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]
print(outputs)
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]

title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
article = "This app is CI/CD for Machine Learning. It automates training, evaluation, and deployment of models to Hugging Face using GitHub Actions."

gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch(share=True)