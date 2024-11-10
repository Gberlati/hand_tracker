from transformers import YolosConfig, YolosModel, AutoImageProcessor
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = YolosModel.from_pretrained("hustvl/yolos-small")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))
print('--------------------------')
pimport cv2
import torch
from transformers import YolosForObjectDetection, AutoImageProcessor
import numpy as np

def main():
    # Load model and processor
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
    
    # Set up webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB (YOLOS expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare image for model
        inputs = image_processor(rgb_frame, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Process results
        probas = outputs.logits.softmax(-1)[0, :, :]
        keep = probas.max(-1).values > 0.9  # Confidence threshold
        
        # Convert predictions to bounding boxes
        target_sizes = torch.tensor([frame.shape[:2]]).to(device)
        results = image_processor.post_process_object_detection(
            outputs, 
            threshold=0.9, 
            target_sizes=target_sizes
        )[0]

        # Draw boxes for detected hands
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            
            # Check if the detected object is a person (which includes hands)
            # COCO dataset label 1 is person
            if label == 1 and score > 0.9:
                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2
                )
                
                # Add label and score
                label_text = f"Hand: {score:.2f}"
                cv2.putText(
                    frame,
                    label_text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Show frame
        cv2.imshow('Hand Detection', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()rint(outputs)
