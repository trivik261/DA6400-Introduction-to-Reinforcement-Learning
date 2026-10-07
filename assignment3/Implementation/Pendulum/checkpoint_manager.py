import os
import torch
import json
from datetime import datetime
class CheckpointManager:
    """
    Manages model checkpoints with automatic saving and loading.
    """
    def __init__(self, checkpoint_dir, keep_best_n=3, keep_last_n=2):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
            keep_last_n: Number of most recent checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_best_n = keep_best_n
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_checkpoints = []
        self.last_checkpoints = []
        self.checkpoint_history = []
    def save_checkpoint(self, agent, timestep, eval_reward, is_best=False, metadata=None):
        """
        Save a checkpoint.
        Args:
            agent: SAC agent to save
            timestep: Current training timestep
            eval_reward: Evaluation reward at this checkpoint
            is_best: Whether this is the best checkpoint so far
            metadata: Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            checkpoint_name = f"best_t{timestep}_r{eval_reward:.1f}_{timestamp}.pt"
        else:
            checkpoint_name = f"checkpoint_t{timestep}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        checkpoint_data = {
            'timestep': timestep,
            'eval_reward': eval_reward,
            'timestamp': timestamp,
            'is_best': is_best,
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'actor': agent.actor.state_dict(),
            'critic_optimizer': agent.critic_optimizer.state_dict(),
            'actor_optimizer': agent.actor_optimizer.state_dict(),
            'log_alpha': agent.log_alpha if agent.auto_entropy_tuning else None,
            'alpha_optimizer': agent.alpha_optimizer.state_dict() if agent.auto_entropy_tuning else None,
            'metadata': metadata or {}
        }
        torch.save(checkpoint_data, checkpoint_path)
        checkpoint_info = {
            'path': checkpoint_path,
            'name': checkpoint_name,
            'timestep': timestep,
            'eval_reward': float(eval_reward),
            'is_best': bool(is_best)
        }
        self.checkpoint_history.append(checkpoint_info)
        if is_best:
            self.best_checkpoints.append(checkpoint_info)
            self.best_checkpoints.sort(key=lambda x: x['eval_reward'], reverse=True)
            if len(self.best_checkpoints) > self.keep_best_n:
                removed = self.best_checkpoints.pop()
                if os.path.exists(removed['path']):
                    os.remove(removed['path'])
                    print(f"[Checkpoint] Removed old best checkpoint: {removed['name']}")
        self.last_checkpoints.append(checkpoint_info)
        if len(self.last_checkpoints) > self.keep_last_n:
            removed = self.last_checkpoints.pop(0)
            if not removed['is_best'] and os.path.exists(removed['path']):
                os.remove(removed['path'])
                print(f"[Checkpoint] Removed old checkpoint: {removed['name']}")
        print(f"[Checkpoint] Saved: {checkpoint_name} (Reward: {eval_reward:.2f})")
        self._save_checkpoint_log()
        return checkpoint_path
    def load_checkpoint(self, agent, checkpoint_path=None, load_best=False):
        """
        Load a checkpoint.
        Args:
            agent: SAC agent to load into
            checkpoint_path: Specific checkpoint path to load
            load_best: Load the best checkpoint if True
        Returns:
            Loaded checkpoint data
        """
        if load_best and self.best_checkpoints:
            checkpoint_path = self.best_checkpoints[0]['path']
        elif checkpoint_path is None and self.last_checkpoints:
            checkpoint_path = self.last_checkpoints[-1]['path']
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"[Checkpoint] No checkpoint found at {checkpoint_path}")
            return None
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.critic_target.load_state_dict(checkpoint['critic_target'])
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if agent.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            agent.log_alpha = checkpoint['log_alpha']
            agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        print(f"[Checkpoint] Loaded: {os.path.basename(checkpoint_path)}")
        print(f"[Checkpoint] Timestep: {checkpoint['timestep']}, Reward: {checkpoint['eval_reward']:.2f}")
        return checkpoint
    def _save_checkpoint_log(self):
        """Save checkpoint history to JSON."""
        log_path = os.path.join(self.checkpoint_dir, 'checkpoint_log.json')
        log_data = {
            'checkpoint_history': self.checkpoint_history,
            'best_checkpoints': self.best_checkpoints,
            'last_checkpoints': self.last_checkpoints
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
    def get_best_checkpoint_path(self):
        """Get path to the best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0]['path']
        return None
    def list_checkpoints(self):
        """List all available checkpoints."""
        print("\n" + "="*70)
        print("CHECKPOINT HISTORY")
        print("="*70)
        if not self.checkpoint_history:
            print("No checkpoints found.")
        else:
            print(f"{'Timestep':<12} {'Reward':<12} {'Best':<8} {'Name':<40}")
            print("-"*70)
            for ckpt in self.checkpoint_history:
                best_marker = "BEST" if ckpt['is_best'] else ""
                print(f"{ckpt['timestep']:<12} {ckpt['eval_reward']:<12.2f} {best_marker:<8} {ckpt['name']:<40}")
        print("="*70 + "\n")
