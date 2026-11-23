
import onnxruntime as ort
import numpy as np
import json
from PIL import Image
from pathlib import Path


class CropDiseasePredictor:

    def __init__(self, model_path, labels_path):

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load class labels
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        print("Crop Disease Model Loaded Successfully!")
        print("Total Classes:", len(self.labels))


    def preprocess(self, image_path):

        # Read image
        image = Image.open(image_path).convert("RGB")

        # Resize image
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to NumPy array
        image = np.array(image, dtype=np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std

        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)


    def predict(self, image_path, top_k=3):

        # Preprocess image
        input_data = self.preprocess(image_path)

        # ONNX inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )[0]

        # Convert logits to probabilities
        logits = output[0]

        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / np.sum(exp_scores)

        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "disease": self.labels[index],
                "confidence": round(
                    float(probabilities[index] * 100), 2
                )
            })

        return results


# Example usage
if __name__ == "__main__":

    MODEL_PATH = "crop_disease_model.onnx"
    LABELS_PATH = "labels.json"

    IMAGE_PATH = "test_image.jpg"

    predictor = CropDiseasePredictor(
        MODEL_PATH,
        LABELS_PATH
    )

    predictions = predictor.predict(IMAGE_PATH)

    print("\nPrediction Results:")

    for i, result in enumerate(predictions, 1):

        print(f"\nRank {i}")
        print("Disease:", result["disease"])
        print("Confidence:", result["confidence"], "%")
