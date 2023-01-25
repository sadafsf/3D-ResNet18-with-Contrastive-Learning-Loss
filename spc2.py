
from pytorch_metric_learning import losses
import torch

class SupConLoss(losses.Loss):
    def __init__(self, temperature=0.1, name=None):
        super(SupConLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        
        # Compute logits
        logits = torch.divide(
            torch.matmul(
                feature_vectors, torch.transpose(feature_vectors)
            ),
            self.temperature,
        )
        return losses.npairs_loss(torch.squeeze(labels), logits)