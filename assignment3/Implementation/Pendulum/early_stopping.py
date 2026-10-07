import numpy as np
class EarlyStopping:
    """
    Early stopping handler to stop training when performance plateaus.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience: Number of evaluations to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    def __call__(self, score, epoch):
        """
        Check if training should stop.
        Args:
            score: Current metric value
            epoch: Current epoch/evaluation number
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            if self.verbose:
                print(f"[Early Stopping] Improvement detected: {self.best_score:.2f} -> {score:.2f}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[Early Stopping] No improvement for {self.counter}/{self.patience} evaluations")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[Early Stopping] Stopping training! Best score: {self.best_score:.2f} at evaluation {self.best_epoch}")
                return True
        return False
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
