import os
import logging
import time
from pathlib import Path
import sys
import warnings
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoProcessor,
    AutoFeatureExtractor
)
from datasets import load_dataset
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import json
from collections import defaultdict

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress warnings
warnings.filterwarnings('ignore')

def get_free_disk_space(path):
    """Get free disk space in GB for the given path."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free // (2**30)  # Convert to GB
    except Exception as e:
        logging.error(f"Error checking disk space: {str(e)}")
        return 0

def check_disk_space(path, required_gb=3):
    """Check if there's enough disk space available."""
    free_gb = get_free_disk_space(path)
    logging.info(f"Free disk space: {free_gb}GB")
    
    if free_gb < required_gb:
        logging.warning(f"Low disk space warning: {free_gb}GB available, {required_gb}GB recommended")
        # Instead of raising an error, return False to allow the script to continue
        return False
    return True

def get_cache_dir():
    """Get the cache directory, trying multiple locations."""
    # Try user's home directory first
    home_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    if os.path.exists(home_cache) and check_disk_space(home_cache, 3):
        return home_cache
        
    # Try current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if check_disk_space(current_dir, 3):
        cache_dir = os.path.join(current_dir, "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
        
    # Try temp directory
    temp_dir = os.path.join(os.environ.get('TEMP', '/tmp'), "huggingface_cache")
    if check_disk_space(temp_dir, 3):
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
        
    # If no suitable directory found, use current directory anyway
    logging.warning("No directory with ideal space found. Using current directory with limited space.")
    cache_dir = os.path.join(current_dir, "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    logging.error(f"Error importing required packages: {str(e)}")
    sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'PIL': 'Pillow',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        logging.error(f"Missing required packages: {', '.join(missing_packages)}")
        logging.error("Please install them using: pip install " + " ".join(missing_packages))
        return False
    return True

def initialize_environment():
    """Initialize the environment and check requirements."""
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            raise RuntimeError("Python 3.7 or higher is required")
            
        # Check dependencies
        if not check_dependencies():
            raise RuntimeError("Missing required dependencies")
            
        # Check disk space
        if not check_disk_space():
            raise RuntimeError("Insufficient disk space")
            
        logging.info("Environment initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error initializing environment: {str(e)}")
        return False

class MedicalTermProcessor:
    def __init__(self):
        self.medical_terms = {
            'anatomy': ['organ', 'tissue', 'muscle', 'bone', 'nerve', 'vessel', 'gland'],
            'pathology': ['disease', 'infection', 'inflammation', 'tumor', 'lesion', 'ulcer'],
            'procedure': ['surgery', 'biopsy', 'endoscopy', 'colonoscopy', 'gastroscopy'],
            'symptom': ['pain', 'bleeding', 'nausea', 'vomiting', 'fever', 'fatigue'],
            'diagnosis': ['diagnosis', 'finding', 'result', 'assessment', 'evaluation']
        }
        
    def extract_terms(self, text):
        """Extract medical terms and their categories from text."""
        found_terms = defaultdict(list)
        text = text.lower()
        
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term in text:
                    found_terms[category].append(term)
                    
        return dict(found_terms)
        
    def get_medical_context(self, text):
        """Get medical context based on terms found."""
        terms = self.extract_terms(text)
        if not terms:
            return "general"
            
        # Return the category with the most terms
        return max(terms.items(), key=lambda x: len(x[1]))[0]

class AdvancedVQADataset:
    def __init__(self, split="raw", max_samples=1000, cache_dir=None):
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "dataset_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.medical_processor = MedicalTermProcessor()
        
        # Check disk space before starting
        if not check_disk_space(self.cache_dir, 5):
            raise RuntimeError("Insufficient disk space for dataset processing")
            
        self._create_metadata()
        
    def _create_metadata(self):
        """Create metadata with optimized processing."""
        try:
            logging.info(f"Loading dataset split: {self.split}")
            logging.info(f"Cache directory: {self.cache_dir}")
            
            dataset_info = load_dataset(
                "SimulaMet-HOST/Kvasir-VQA",
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=True
            )
            
            self.metadata = []
            progress_bar = tqdm(total=self.max_samples, desc="Loading dataset")
            
            for i, item in enumerate(dataset_info):
                if i >= self.max_samples:
                    break
                    
                try:
                    if not self._validate_sample(item):
                        continue
                        
                    processed_item = self._process_sample(item)
                    if processed_item:
                        self.metadata.append(processed_item)
                        progress_bar.update(1)
                        
                except Exception as e:
                    logging.warning(f"Error processing sample {i}: {str(e)}")
                    continue
                    
            progress_bar.close()
            logging.info(f"Successfully processed {len(self.metadata)} samples")
            
        except Exception as e:
            logging.error(f"Error loading dataset info: {str(e)}")
            raise
            
    def _validate_sample(self, item):
        """Enhanced sample validation."""
        required_fields = ['image_url', 'question', 'answer']
        return all(item.get(field) for field in required_fields)
        
    def _process_sample(self, item):
        """Process and enhance sample data."""
        try:
            if not self._validate_sample(item):
                return None
                
            # Extract medical context
            medical_context = self.medical_processor.get_medical_context(
                item['question'] + ' ' + item['answer']
            )
            
            # Enhanced data structure
            return {
                'image_url': item['image_url'],
                'question': item['question'],
                'answer': item['answer'],
                'metadata': {
                    'question_type': self._classify_question_type(item['question']),
                    'answer_length': len(item['answer'].split()),
                    'medical_context': medical_context,
                    'medical_terms': self.medical_processor.extract_terms(
                        item['question'] + ' ' + item['answer']
                    )
                }
            }
        except Exception as e:
            logging.warning(f"Error processing sample: {str(e)}")
            return None
            
    def _classify_question_type(self, question):
        """Classify the type of question."""
        question = question.lower()
        if 'what' in question:
            return 'what'
        elif 'where' in question:
            return 'where'
        elif 'how' in question:
            return 'how'
        elif 'why' in question:
            return 'why'
        elif 'when' in question:
            return 'when'
        else:
            return 'other'
            
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        try:
            item = self.metadata[idx]
            
            # Load and process image with error handling
            image = self._load_image(item['image_url'])
            if image is None:
                raise ValueError(f"Failed to load image: {item['image_url']}")
                
            return {
                'image': image,
                'question': item['question'],
                'answer': item['answer'],
                'metadata': item['metadata']
            }
            
        except Exception as e:
            logging.error(f"Error loading item {idx}: {str(e)}")
            raise
            
    def _load_image(self, url):
        """Load image with enhanced error handling."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            logging.warning(f"Error loading image {url}: {str(e)}")
            return None

class SmartVQAModel(nn.Module):
    def __init__(self, model_name="microsoft/git-base"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            local_files_only=False,
            resume_download=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Add confidence scoring layer
        self.confidence_layer = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, images, questions):
        inputs = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        outputs = self.model(**inputs)
        
        # Calculate confidence scores
        confidence_scores = self.confidence_layer(outputs.last_hidden_state[:, 0, :])
        
        return outputs, confidence_scores
        
    def generate_answer(self, image, question):
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        )
        
        # Get attention weights
        outputs = self.model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            output_attentions=True,
            return_dict_in_generate=True
        )
        
        # Calculate confidence score
        confidence = self.confidence_layer(outputs.sequences[:, 0, :]).item()
        
        # Get attention weights for visualization
        attention_weights = outputs.attentions[-1][0].mean(dim=1)
        
        return {
            'answer': self.processor.decode(outputs.sequences[0], skip_special_tokens=True),
            'confidence': confidence,
            'attention_weights': attention_weights
        }
        
    def visualize_attention(self, attention_weights, question, answer):
        """Visualize attention weights."""
        plt.figure(figsize=(10, 5))
        plt.imshow(attention_weights.detach().cpu().numpy())
        plt.title(f"Attention Visualization\nQ: {question}\nA: {answer}")
        plt.colorbar()
        plt.savefig('attention_visualization.png')
        plt.close()

class ModelTrainer:
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=5e-5)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=len(train_loader) // 10,
            num_training_steps=len(train_loader) * 10
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_confidence = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/10"):
            self.optimizer.zero_grad()
            
            # Move batch to device
            images = batch['image'].to(self.device)
            questions = batch['question']
            
            # Forward pass
            outputs, confidence_scores = self.model(images, questions)
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            total_confidence += confidence_scores.mean().item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
        return total_loss / len(self.train_loader), total_confidence / len(self.train_loader)
        
    def save_checkpoint(self, epoch, loss, confidence):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'confidence': confidence
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save training metrics
        metrics = {
            'epoch': epoch + 1,
            'loss': loss,
            'confidence': confidence,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('training_metrics.json', 'a') as f:
            json.dump(metrics, f)
            f.write('\n')

def train_model():
    """Main training function with advanced features."""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Initialize model
        logging.info("Initializing model...")
        model = SmartVQAModel()
        model = model.to(device)
        
        # Load dataset
        logging.info("Loading dataset...")
        train_dataset = AdvancedVQADataset(
            split="raw",
            max_samples=1000
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4
        )
        
        # Initialize trainer
        trainer = ModelTrainer(model, train_loader, device)
        
        # Training loop
        logging.info("Starting training...")
        for epoch in range(10):
            loss, confidence = trainer.train_epoch(epoch)
            logging.info(f"Epoch {epoch + 1} - Loss: {loss:.4f}, Confidence: {confidence:.4f}")
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, loss, confidence)
            
            # Visualize attention for a sample
            if epoch % 2 == 0:  # Every 2 epochs
                sample = next(iter(train_loader))
                result = model.generate_answer(
                    sample['image'][0].to(device),
                    sample['question'][0]
                )
                model.visualize_attention(
                    result['attention_weights'],
                    sample['question'][0],
                    result['answer']
                )
                
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

# ... rest of existing code ... 