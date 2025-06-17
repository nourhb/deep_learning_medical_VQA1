import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
import torch
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_dir: str = MODEL_CONFIG["model_dir"]):
        self.model_dir = model_dir
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.current_model: Optional[str] = None
        self._load_metadata()

    def _load_metadata(self):
        """Load model metadata from JSON file."""
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.model_metadata = json.load(f)

    def _save_metadata(self):
        """Save model metadata to JSON file."""
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2)

    def load_model(self, model_name: str) -> bool:
        """Load a specific model version."""
        try:
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                logger.error(f"Model {model_name} not found at {model_path}")
                return False

            # Load PyTorch model
            model = torch.load(model_path)
            model.eval()  # Set to evaluation mode
            self.models[model_name] = model
            self.current_model = model_name
            logger.info(f"Loaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False

    def save_model(self, model: torch.nn.Module, version: str, metadata: Dict = None) -> bool:
        """Save a new model version with metadata."""
        try:
            # Create version directory
            version_dir = os.path.join(self.model_dir, version)
            os.makedirs(version_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(version_dir, "model.pt")
            torch.save(model.state_dict(), model_path)

            # Save metadata
            model_metadata = {
                "version": version,
                "created_at": datetime.now().isoformat(),
                "metrics": {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "layers": len(list(model.modules())),
                }
            }
            if metadata:
                model_metadata.update(metadata)

            self.model_metadata[version] = model_metadata
            self._save_metadata()

            logger.info(f"Saved model version {version}")
            return True
        except Exception as e:
            logger.error(f"Error saving model version {version}: {str(e)}")
            return False

    def get_model_versions(self) -> List[Dict]:
        """Get list of available model versions with metadata."""
        return [
            {
                "version": version,
                **metadata
            }
            for version, metadata in self.model_metadata.items()
        ]

    def get_current_model(self) -> Optional[torch.nn.Module]:
        """Get the currently loaded model."""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]
        return None

    def get_model_metrics(self, version: str) -> Optional[Dict]:
        """Get metrics for a specific model version."""
        if version in self.model_metadata:
            return self.model_metadata[version].get("metrics")
        return None

    def delete_model(self, version: str) -> bool:
        """Delete a model version."""
        try:
            if version in self.models:
                del self.models[version]
            if version in self.model_metadata:
                del self.model_metadata[version]
                self._save_metadata()

            version_dir = os.path.join(self.model_dir, version)
            if os.path.exists(version_dir):
                import shutil
                shutil.rmtree(version_dir)

            logger.info(f"Deleted model version {version}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model version {version}: {str(e)}")
            return False 