import torch
from PIL import Image
import torchvision.transforms as T
from transformers import BertTokenizer, BertModel, AutoModel
import json
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import timm

# Load label map
with open("label_map_continuous.json", "r") as f:
    answer_to_index = json.load(f)
    index_to_answer = {str(idx): ans for ans, idx in answer_to_index.items()}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class HierarchicalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Hierarchical attention weights
        self.hierarchical_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Multi-scale attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Hierarchical attention weights
        hierarchical_weights = F.softmax(self.hierarchical_weights, dim=0)
        attn = attn * hierarchical_weights.view(1, 1, 1, 1)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x1, x2):
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * x1 + weights[1] * x2
        return self.fusion_layer(torch.cat([fused, x1 * x2], dim=-1))

class KnowledgeDistillation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.teacher_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.student_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, student_features, teacher_features):
        student_proj = self.student_projection(student_features)
        teacher_proj = self.teacher_projection(teacher_features)
        return F.mse_loss(student_proj, teacher_proj)

class CrossModalTransformer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context, mask=None):
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(context.shape[0], -1, self.num_heads, context.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(context.shape[0], -1, self.num_heads, context.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], -1, x.shape[-1])
        x = self.out_proj(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x

class ContrastiveLearning(nn.Module):
    def __init__(self, dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, features1, features2):
        z1 = self.projection(features1)
        z2 = self.projection(features2)
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        logits = torch.matmul(z1, z2.transpose(-2, -1)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

class AdaptiveFeaturePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * attention_weights, dim=1)

class MedicalConfidenceEstimator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return self.confidence_net(features)

class MedicalAttentionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, context):
        gate = self.gate(torch.cat([features, context], dim=-1))
        return features * gate

class MedicalFeatureValidator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.validator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return self.validator(features)

class VQAModel(nn.Module):
    def __init__(self, num_classes, image_size=(384, 384)):
        super().__init__()
        self.image_size = image_size
        
        # Image encoder (EfficientNet-B4)
        self.image_encoder = timm.create_model('efficientnet_b4', pretrained=True)
        self.image_features = self.image_encoder.num_features
        
        # Question encoder (BERT)
        self.question_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.question_features = self.question_encoder.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Feature fusion
        self.fusion = FeatureFusion(self.image_features, self.question_features)
        
        # Hierarchical attention
        self.attention = HierarchicalAttention(self.fusion.output_dim)
        
        # Knowledge distillation
        self.knowledge = KnowledgeDistillation(self.attention.output_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.knowledge.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, question):
        # Process image
        image_features = self.image_encoder.forward_features(images)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Process question
        encoded = self.tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(images.device) for k, v in encoded.items()}
        question_outputs = self.question_encoder(**encoded)
        question_features = question_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Feature fusion
        fused_features = self.fusion(image_features, question_features)
        
        # Apply attention
        attended_features = self.attention(fused_features)
        
        # Apply knowledge distillation
        distilled_features = self.knowledge(attended_features)
        
        # Final classification
        logits = self.classifier(distilled_features)
        
        return logits

class VQALoss(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        
    def forward(self, outputs, labels):
        logits = outputs['logits']
        uncertainty = outputs['uncertainty']
        confidence = outputs['confidence']
        vision_validity = outputs['vision_validity']
        text_validity = outputs['text_validity']
        
        # Uncertainty-aware cross entropy with medical focus
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        weighted_loss = ce_loss * (1 - uncertainty.squeeze())
        
        # Confidence-based weighting
        confidence_weight = confidence.squeeze()
        weighted_loss = weighted_loss * confidence_weight
        
        # Validity regularization
        validity_loss = F.mse_loss(vision_validity, torch.ones_like(vision_validity)) + \
                       F.mse_loss(text_validity, torch.ones_like(text_validity))
        
        # Add label smoothing for medical robustness
        smooth_loss = -torch.mean(torch.log_softmax(logits, dim=-1), dim=-1)
        weighted_loss = 0.9 * weighted_loss + 0.1 * smooth_loss
        
        # Combine losses with medical focus
        total_loss = weighted_loss.mean() + 0.1 * validity_loss
        
        return total_loss

# Initialize model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        model_path = "saved_models/best_vqa_model_cleaned.pth"
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model = VQAModel(num_labels=len(answer_to_index))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(torch.device("cpu")).eval()
        tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Image preprocessing
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.RandomPerspective(distortion_scale=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image, question: str):
    global model, tokenizer
    if model is None:
        load_model()
    
    if image is None or not question.strip():
        return "Image ou question manquante"
    
    image_tensor = train_transform(image).unsqueeze(0).to("cpu")
    tokens = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokens = {key: val.to("cpu") for key, val in tokens.items()}
    
    with torch.no_grad():
        output = model(image_tensor, tokens['input_ids'], tokens['attention_mask'])
        
        # Check confidence threshold
        confidence = output['confidence'].item()
        if confidence < model.confidence_threshold:
            return "⚠️ Confiance insuffisante pour fournir une réponse fiable"
        
        # Get prediction with uncertainty
        logits = output['logits']
        uncertainty = output['uncertainty'].item()
        
        if uncertainty > 0.3:  # High uncertainty threshold
            return "⚠️ Incertitude élevée dans la réponse"
        
        _, pred = torch.max(logits, 1)
        mapped_answer = index_to_answer.get(str(pred.item()), "❓ Inconnu")
        
        # Add confidence level to response
        confidence_level = "Élevée" if confidence > 0.9 else "Moyenne" if confidence > 0.8 else "Faible"
        return f"{mapped_answer} (Confiance: {confidence_level})"