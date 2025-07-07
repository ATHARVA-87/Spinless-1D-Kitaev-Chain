import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor with residual connections.
    
    Extracts common features from input parameters for both
    edge localization and eigenvalue spectrum prediction.
    """
    def __init__(self, input_dim: int = 5, hidden_dim1: int = 256, hidden_dim2: int = 512):
        """
        Initialize the feature extractor.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim1: Dimension of first hidden layer
            hidden_dim2: Dimension of second hidden layer
        """
        super(FeatureExtractor, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.hidden_layer = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(p=0.1)
        
        # Residual connection projection if dimensions don't match
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # GELU activation for better handling of negative values
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Features tensor of shape [batch_size, hidden_dim2]
        """
        # First layer with normalization and dropout
        x1 = self.input_layer(x)
        x1 = self.layer_norm1(x1)
        x1 = self.activation(x1)
        x1 = self.dropout1(x1)
        
        # Second layer with residual connection
        residual = self.residual_proj(x1)
        x2 = self.hidden_layer(x1)
        x2 = self.layer_norm2(x2)
        x2 = self.activation(x2)
        x2 = self.dropout2(x2)
        
        # Add residual connection
        output = x2 + residual
        
        return output


class EdgeLocalizationHead(nn.Module):
    """
    Specialized head for edge localization prediction.
    """
    def __init__(self, in_features: int = 512, hidden_dim: int = 128):
        """
        Initialize the edge localization head.
        
        Args:
            in_features: Number of input features from the feature extractor
            hidden_dim: Dimension of hidden layer in the head
        """
        super(EdgeLocalizationHead, self).__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the edge localization head.
        
        Args:
            x: Input tensor from feature extractor [batch_size, in_features]
            
        Returns:
            Edge localization prediction [batch_size, 1]
        """
        residual = x  # Save input for residual connection
        
        # Project to lower dimension for localization task
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x


class EigenvalueSpectrumHead(nn.Module):
    """
    Specialized head for eigenvalue spectrum prediction with optional 1D convolution for
    better capturing spectral correlations.
    """
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, output_dim: int = 180, 
                 use_conv: bool = True):
        """
        Initialize the eigenvalue spectrum head.
        
        Args:
            in_features: Number of input features from the feature extractor
            hidden_dim: Dimension of hidden layer in the head
            output_dim: Number of eigenvalues to predict
            use_conv: Whether to use 1D convolution for sequence modeling
        """
        super(EigenvalueSpectrumHead, self).__init__()
        
        self.use_conv = use_conv
        
        # First projection from feature extractor
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        # Optional 1D convolution for sequence modeling
        if use_conv:
            # Reshape features to [batch, channels, sequence_length]
            self.reshape_dim = 16
            assert hidden_dim % self.reshape_dim == 0, "hidden_dim must be divisible by reshape_dim"
            self.seq_length = hidden_dim // self.reshape_dim
            
            # 1D convolution with residual connection
            self.conv1d = nn.Conv1d(self.reshape_dim, self.reshape_dim, kernel_size=3, padding=1)
            self.layer_norm_conv = nn.LayerNorm([self.reshape_dim, self.seq_length])
        
        # Output projection
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the eigenvalue spectrum head.
        
        Args:
            x: Input tensor from feature extractor [batch_size, in_features]
            
        Returns:
            Eigenvalue spectrum prediction [batch_size, output_dim]
        """
        # Initial projection
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        
        # Optional 1D convolution for sequence modeling
        if self.use_conv:
            batch_size = x.shape[0]
            # Reshape to [batch, channels, sequence_length]
            x_reshaped = x.view(batch_size, self.reshape_dim, self.seq_length)
            
            # Apply convolution with residual connection
            residual = x_reshaped
            x_conv = self.conv1d(x_reshaped)
            x_conv = self.layer_norm_conv(x_conv)
            x_conv = self.activation(x_conv)
            x_conv = x_conv + residual
            
            # Reshape back
            x = x_conv.reshape(batch_size, -1)
        
        # Output projection
        x = self.fc2(x)
        
        return x


class EnhancedBdGPredictor(nn.Module):
    """
    Enhanced MLP model for predicting edge localization and BdG eigenvalue spectrum.
    
    Features:
    - Shared feature extractor with residual connections
    - Specialized prediction heads for each task
    - Layer normalization and dropout for improved training stability
    - Optional 1D convolution for spectral modeling
    
    Input: 5D features [L, mu, t, delta, disorder_strength]
    Output: 181D (1 for edge_localization, 180 for eigenvalues)
    """
    def __init__(self, input_dim: int = 5, hidden_dim1: int = 256, 
                 hidden_dim2: int = 512, output_dim: int = 181,
                 use_conv: bool = True, dropout_rate: float = 0.1):
        """
        Initialize EnhancedBdGPredictor model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim1: Dimension of first hidden layer
            hidden_dim2: Dimension of second hidden layer (feature extractor output)
            output_dim: Total dimension of output (edge_localization + eigenvalues)
            use_conv: Whether to use 1D convolution in eigenvalue head
            dropout_rate: Dropout probability for regularization
        """
        super(EnhancedBdGPredictor, self).__init__()
        
        # Save configuration
        self.output_dim = output_dim
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2
        )
        
        # Specialized prediction heads
        self.edge_loc_head = EdgeLocalizationHead(
            in_features=hidden_dim2,
            hidden_dim=hidden_dim1 // 2
        )
        
        self.eigenvalue_head = EigenvalueSpectrumHead(
            in_features=hidden_dim2,
            hidden_dim=hidden_dim2,
            output_dim=output_dim - 1,  # Subtract edge localization output
            use_conv=use_conv
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim] where the first element
            is the edge localization prediction and the rest are eigenvalues
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Get predictions from specialized heads
        edge_loc = self.edge_loc_head(features)
        eigenvalues = self.eigenvalue_head(features)
        
        # Concatenate predictions along dimension 1
        output = torch.cat([edge_loc, eigenvalues], dim=1)
        
        return output


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Compute Huber loss (smooth L1).
    
    Huber loss is less sensitive to outliers than MSE:
    - For error < delta: 0.5 * error^2
    - For error >= delta: delta * (|error| - 0.5 * delta)
    
    Args:
        pred: Predicted values
        target: Target values
        delta: Threshold for switching between MSE and MAE
        
    Returns:
        Huber loss value
    """
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    return loss


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute spectral loss by comparing sorted predicted and target spectra.
    
    This loss encourages the model to predict the correct eigenvalue distribution
    regardless of the specific ordering of eigenvalues.
    
    Args:
        pred: Predicted eigenvalues [batch_size, n_eigenvalues]
        target: Target eigenvalues [batch_size, n_eigenvalues]
        
    Returns:
        Spectral loss value
    """
    # Sort both predicted and target eigenvalues
    sorted_pred, _ = torch.sort(pred, dim=1)
    sorted_target, _ = torch.sort(target, dim=1)
    
    # Compute MSE between sorted spectra
    return F.mse_loss(sorted_pred, sorted_target)


def enhanced_masked_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor, 
    lambda_eig: float = 1.0,
    lambda_spectral: float = 0.2,
    huber_delta: float = 1.0,
    use_huber: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute enhanced loss for edge localization and eigenvalues.
    
    Features:
    - Optional Huber loss for eigenvalues to reduce impact of outliers
    - Additional spectral loss component for global spectrum shape
    - Improved mask handling and normalization
    
    Args:
        pred: Predicted values [batch_size, 181]
        target: Target values [batch_size, 181]
        mask: Mask tensor [batch_size, 180]
        lambda_eig: Weight for eigenvalue loss
        lambda_spectral: Weight for spectral loss component
        huber_delta: Delta parameter for Huber loss
        use_huber: Whether to use Huber loss instead of MSE for eigenvalues
        
    Returns:
        total_loss, edge_loc_loss, eig_loss
    """
    # Input validation
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    assert mask.shape[1] == pred.shape[1] - 1, f"Mask shape {mask.shape} incompatible with pred {pred.shape}"
    
    # Split predictions and targets
    edge_loc_pred = pred[:, 0].view(-1, 1)
    eig_pred = pred[:, 1:]
    
    edge_loc_target = target[:, 0].view(-1, 1)
    eig_target = target[:, 1:]
    
    # Compute edge localization loss
    edge_loc_loss = F.mse_loss(edge_loc_pred, edge_loc_target)
    
    # Compute masked eigenvalue loss
    # Apply mask to predictions and targets
    masked_eig_pred = eig_pred * mask
    masked_eig_target = eig_target * mask
    
    # Compute loss only for unmasked values
    if use_huber:
        eig_error = huber_loss(masked_eig_pred, masked_eig_target, delta=huber_delta)
    else:
        eig_error = (masked_eig_pred - masked_eig_target) ** 2
    
    # Normalize by sum of mask to account for different valid lengths
    mask_sum = torch.sum(mask, dim=1, keepdim=True)
    # Avoid division by zero with safe normalization
    safe_mask_sum = torch.clamp(mask_sum, min=1.0)
    eig_loss = torch.mean(torch.sum(eig_error, dim=1) / safe_mask_sum.squeeze())
    
    # Compute spectral loss only for entries with sufficient non-masked values
    # We only compute this for samples with at least half the spectrum available
    valid_samples = mask_sum.squeeze() >= (mask.shape[1] / 2)
    
    if torch.any(valid_samples):
        # Extract valid samples for spectral loss
        valid_pred = eig_pred[valid_samples]
        valid_target = eig_target[valid_samples]
        valid_mask = mask[valid_samples]
        
        # Fill masked values with mean of unmasked values to make sorting meaningful
        # For each sample in the batch:
        spec_loss = 0.0
        for i in range(valid_pred.shape[0]):
            # Get unmasked values
            unmasked_pred = valid_pred[i][valid_mask[i] > 0]
            unmasked_target = valid_target[i][valid_mask[i] > 0]
            
            # Sort unmasked values
            sorted_pred, _ = torch.sort(unmasked_pred)
            sorted_target, _ = torch.sort(unmasked_target)
            
            # Compute MSE between sorted spectra of equal length
            sample_spec_loss = F.mse_loss(sorted_pred, sorted_target)
            spec_loss += sample_spec_loss
            
        # Average over valid samples
        spec_loss = spec_loss / valid_samples.sum()
    else:
        # No valid samples for spectral loss
        spec_loss = torch.tensor(0.0, device=pred.device)
    
    # Combine losses
    total_loss = edge_loc_loss + lambda_eig * eig_loss + lambda_spectral * spec_loss
    
    return total_loss, edge_loc_loss, eig_loss


def compute_accuracy_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.5,
    eigenvalue_tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Compute accuracy metrics for model evaluation.
    
    Args:
        pred: Predicted values [batch_size, 181]
        target: Target values [batch_size, 181]
        mask: Optional mask tensor [batch_size, 180] for eigenvalues
        threshold: Threshold for binary classification of edge localization
        eigenvalue_tolerance: Relative tolerance for eigenvalue accuracy
        
    Returns:
        Dictionary containing accuracy metrics:
        - edge_binary_accuracy: Binary classification accuracy for edge localization
        - edge_mae: Mean absolute error for edge localization
        - eigenvalue_accuracy: Percentage of eigenvalues within tolerance
        - overall_accuracy: Combined metric
    """
    metrics = {}
    
    # Edge localization binary accuracy
    edge_pred = pred[:, 0]
    edge_target = target[:, 0]
    
    # Binary classification accuracy (treating edge_loc as probability)
    edge_pred_binary = (edge_pred > threshold).float()
    edge_target_binary = (edge_target > threshold).float()
    edge_binary_accuracy = (edge_pred_binary == edge_target_binary).float().mean().item()
    metrics['edge_binary_accuracy'] = edge_binary_accuracy
    
    # Edge localization mean absolute error
    edge_mae = torch.abs(edge_pred - edge_target).mean().item()
    metrics['edge_mae'] = edge_mae
    
    # Eigenvalue accuracy - percentage within tolerance
    if mask is not None:
        eig_pred = pred[:, 1:]
        eig_target = target[:, 1:]
        
        # Apply mask
        masked_eig_pred = eig_pred * mask
        masked_eig_target = eig_target * mask
        
        # Calculate relative error with threshold for each eigenvalue
        relative_error = torch.abs(masked_eig_pred - masked_eig_target)
        # Add small epsilon to avoid division by zero
        target_abs = torch.abs(masked_eig_target) + 1e-8
        relative_error = relative_error / target_abs
        
        # Count values within tolerance where mask is applied
        within_tolerance = (relative_error < eigenvalue_tolerance) & (mask > 0)
        total_unmasked = torch.sum(mask > 0).item()
        
        if total_unmasked > 0:
            eigenvalue_accuracy = torch.sum(within_tolerance).item() / total_unmasked
        else:
            eigenvalue_accuracy = 0.0
            
        metrics['eigenvalue_accuracy'] = eigenvalue_accuracy
        
        # Overall accuracy (weighted combination)
        metrics['overall_accuracy'] = 0.5 * edge_binary_accuracy + 0.5 * eigenvalue_accuracy
    else:
        # If no mask provided, only report edge metrics
        metrics['overall_accuracy'] = edge_binary_accuracy
    
    return metrics


# For backward compatibility, maintain the original function signature
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, 
                   mask: torch.Tensor, lambda_eig: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute masked MSE loss for edge localization and eigenvalues.
    Maintained for backward compatibility.
    
    Args:
        pred: Predicted values [batch_size, 181]
        target: Target values [batch_size, 181]
        mask: Mask tensor [batch_size, 180]
        lambda_eig: Weight for eigenvalue loss
        
    Returns:
        total_loss, edge_loc_loss, eig_loss
    """
    return enhanced_masked_loss(
        pred=pred,
        target=target,
        mask=mask,
        lambda_eig=lambda_eig,
        lambda_spectral=0.0,  # Disable spectral loss for compatibility
        use_huber=False  # Use MSE for compatibility
    )


# For convenience, create an alias to the original BdGPredictor
# This allows for drop-in replacement with the enhanced model
BdGPredictor = EnhancedBdGPredictor