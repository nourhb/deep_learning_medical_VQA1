import torch
import torch.nn as nn
import torch.optim as optim
from model import VQAModel
import json
import os

def create_basic_model():
    print("Creating basic VQA model...")
    
    # Create saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    try:
        # Load label map
        with open("label_map_continuous.json", "r") as f:
            label_map = json.load(f)
        
        # Initialize model
        model = VQAModel(num_labels=len(label_map))
        
        # Create a simple optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Save the model
        model_path = "saved_models/best_vqa_model_cleaned.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        
        print(f"Basic model saved successfully to {model_path}!")
        return True
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return False

if __name__ == "__main__":
    create_basic_model() 