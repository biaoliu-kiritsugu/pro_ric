import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class ScoreClassifier(nn.Module):
    def __init__(self, model_name, num_rewards=4, hidden_size=1024, reward_stats=None):
        """
        Initialize the score classifier.
        
        Args:
            model_name: Name of the pretrained model to use
            num_rewards: Number of reward scores to predict
            hidden_size: Size of the hidden layer
            reward_stats: Dictionary containing mean and std for each reward score
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        #for param in self.model.parameters():
            #param.requires_grad = False
        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_rewards),
            nn.Softmax(dim=1)
        )
        # Store reward statistics for normalization
        if reward_stats is not None:
            self.register_buffer('reward_means', torch.tensor(reward_stats[:, 0], dtype=torch.float32))
            self.register_buffer('reward_stds', torch.tensor(reward_stats[:, 1], dtype=torch.float32))
        else:
            self.reward_means = None
            self.reward_stds = None
        
    def forward(self, prompts):
        """
        Forward pass of the model.
        
        Args:
            prompts: List of text prompts or tensor of tokenized inputs
            
        Returns:
            Tensor of predicted scores (normalized)
        """
        # If input is text, tokenize it
        if isinstance(prompts[0], str):
            inputs = self.tokenizer(prompts, padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = prompts
            
        # Get embeddings from the base model
        outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Get raw scores from the classifier head
        #raw_scores = self.classifier(pooled_output)
        scores = self.classifier(pooled_output)
        
        # Normalize scores if statistics are available
        """
        if self.reward_means is not None and self.reward_stds is not None:
            scores = (raw_scores - self.reward_means) / self.reward_stds
        else:
            raise ValueError("reward stats informtin not found")
        """
        #scores=torch.nn.functional.softmax(scores,dim=1)
        return scores
    
    def predict(self, prompts):
        """
        Make predictions for a list of prompts.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Numpy array of predicted scores (normalized)
        """
        self.eval()
        with torch.no_grad():
            scores = self(prompts)
            return scores.cpu().numpy() 