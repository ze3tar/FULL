#!/usr/bin/env python3
"""
LSTM-based Obstacle Motion Predictor for Dynamic Path Planning
Predicts future obstacle positions based on movement history
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import time


class ObstacleTrajectoryDataset(Dataset):
    """Dataset for obstacle trajectory sequences"""
    
    def __init__(self, trajectories, history_len=10, predict_len=3):
        """
        Args:
            trajectories: List of obstacle trajectories [(t, x, y, z, vx, vy, vz), ...]
            history_len: Number of past steps to use
            predict_len: Number of future steps to predict
        """
        self.history_len = history_len
        self.predict_len = predict_len
        self.samples = []
        
        # Process trajectories into training samples
        for traj in trajectories:
            if len(traj) < history_len + predict_len:
                continue
            
            for i in range(len(traj) - history_len - predict_len + 1):
                history = traj[i:i+history_len]  # (history_len, 7)
                future = traj[i+history_len:i+history_len+predict_len]  # (predict_len, 7)
                
                # Extract positions and velocities
                hist_pos = history[:, 1:4]  # (history_len, 3) - x, y, z
                hist_vel = history[:, 4:7]  # (history_len, 3) - vx, vy, vz
                future_pos = future[:, 1:4]  # (predict_len, 3)
                
                self.samples.append((
                    np.hstack([hist_pos, hist_vel]),  # (history_len, 6)
                    future_pos  # (predict_len, 3)
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        history, future = self.samples[idx]
        return (
            torch.FloatTensor(history),
            torch.FloatTensor(future.reshape(-1))  # Flatten future positions
        )


class ObstaclePredictorLSTM(nn.Module):
    """LSTM network for predicting obstacle future positions"""
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, 
                 predict_steps=3, output_dim=3):
        super(ObstaclePredictorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_steps = predict_steps
        self.output_dim = output_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, predict_steps * output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        Returns:
            predictions: (batch, predict_steps * output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


def generate_training_data(n_trajectories=1000, traj_length=50, 
                          motion_types=['linear', 'circular', 'random']):
    """
    Generate synthetic obstacle trajectories for training
    
    Args:
        n_trajectories: Number of trajectories to generate
        traj_length: Length of each trajectory
        motion_types: Types of motion patterns
    
    Returns:
        trajectories: List of trajectory arrays
    """
    print(f"Generating {n_trajectories} training trajectories...")
    trajectories = []
    
    for i in range(n_trajectories):
        motion_type = np.random.choice(motion_types)
        
        # Initialize
        t = np.arange(traj_length)
        dt = 0.1
        
        if motion_type == 'linear':
            # Constant velocity motion
            start_pos = np.random.uniform(-5, 5, 3)
            velocity = np.random.uniform(-0.5, 0.5, 3)
            
            positions = start_pos + velocity * t[:, np.newaxis]
            velocities = np.tile(velocity, (traj_length, 1))
        
        elif motion_type == 'circular':
            # Circular motion
            center = np.random.uniform(-3, 3, 3)
            radius = np.random.uniform(1, 3)
            angular_vel = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2*np.pi)
            
            theta = angular_vel * t + phase
            positions = center + radius * np.column_stack([
                np.cos(theta),
                np.sin(theta),
                np.zeros_like(theta)
            ])
            
            velocities = radius * angular_vel * np.column_stack([
                -np.sin(theta),
                np.cos(theta),
                np.zeros_like(theta)
            ])
        
        else:  # random walk
            # Random walk with smoothing
            start_pos = np.random.uniform(-5, 5, 3)
            velocities = np.random.normal(0, 0.3, (traj_length, 3))
            velocities = np.cumsum(velocities, axis=0) * 0.1
            
            positions = start_pos + np.cumsum(velocities * dt, axis=0)
        
        # Combine into trajectory
        trajectory = np.column_stack([
            t,
            positions,
            velocities
        ])
        
        trajectories.append(trajectory)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_trajectories}")
    
    return trajectories


def train_predictor(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """
    Train the LSTM predictor
    
    Args:
        model: ObstaclePredictorLSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nTraining on {device}")
    print("="*60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (history, future) in enumerate(train_loader):
            history, future = history.to(device), future.to(device)
            
            optimizer.zero_grad()
            predictions = model(history)
            loss = criterion(predictions, future)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for history, future in val_loader:
                history, future = history.to(device), future.to(device)
                predictions = model(history)
                loss = criterion(predictions, future)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'obstacle_predictor_best.pth')
            print(f"  ✓ New best model saved!")
    
    print("\n" + "="*60)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


class ObstaclePredictor:
    """High-level interface for obstacle prediction"""
    
    def __init__(self, model_path='obstacle_predictor_best.pth', 
                 history_len=10, predict_steps=3, device='cpu'):
        """
        Args:
            model_path: Path to trained model
            history_len: Number of history steps required
            predict_steps: Number of future steps to predict
            device: Device to run inference on
        """
        self.history_len = history_len
        self.predict_steps = predict_steps
        self.device = device
        
        # Load model
        self.model = ObstaclePredictorLSTM(
            input_size=6,
            hidden_size=128,
            num_layers=2,
            predict_steps=predict_steps,
            output_dim=3
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded obstacle predictor from {model_path}")
    
    def predict(self, history):
        """
        Predict future obstacle positions
        
        Args:
            history: Array of shape (history_len, 6) containing [x, y, z, vx, vy, vz]
        
        Returns:
            predictions: Array of shape (predict_steps, 3) containing future [x, y, z]
        """
        if len(history) != self.history_len:
            raise ValueError(f"History must have length {self.history_len}")
        
        # Convert to tensor
        history_tensor = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(history_tensor)
        
        # Reshape and return
        predictions = predictions.cpu().numpy().reshape(self.predict_steps, 3)
        return predictions
    
    def predict_batch(self, histories):
        """
        Predict for multiple obstacles at once
        
        Args:
            histories: Array of shape (n_obstacles, history_len, 6)
        
        Returns:
            predictions: Array of shape (n_obstacles, predict_steps, 3)
        """
        histories_tensor = torch.FloatTensor(histories).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(histories_tensor)
        
        predictions = predictions.cpu().numpy()
        predictions = predictions.reshape(-1, self.predict_steps, 3)
        return predictions


class DynamicObstacleManager:
    """
    Manages dynamic obstacles for real-time planning
    Integrates obstacle tracking, prediction, and replanning triggers
    """
    
    def __init__(self, predictor, prediction_horizon=3):
        """
        Args:
            predictor: ObstaclePredictor instance
            prediction_horizon: How many steps ahead to predict
        """
        self.predictor = predictor
        self.prediction_horizon = prediction_horizon
        
        # Obstacle tracking
        self.obstacles = {}  # {id: {'history': [...], 'predicted': [...]}}
        self.history_len = predictor.history_len
    
    def update_obstacle(self, obstacle_id, position, velocity, timestamp):
        """
        Update obstacle state
        
        Args:
            obstacle_id: Unique obstacle identifier
            position: (x, y, z) position
            velocity: (vx, vy, vz) velocity
            timestamp: Current timestamp
        """
        state = np.hstack([position, velocity])
        
        if obstacle_id not in self.obstacles:
            self.obstacles[obstacle_id] = {
                'history': [state],
                'predicted': None,
                'last_update': timestamp
            }
        else:
            history = self.obstacles[obstacle_id]['history']
            history.append(state)
            
            # Keep only recent history
            if len(history) > self.history_len:
                history = history[-self.history_len:]
            
            self.obstacles[obstacle_id]['history'] = history
            self.obstacles[obstacle_id]['last_update'] = timestamp
            
            # Predict if we have enough history
            if len(history) >= self.history_len:
                predicted = self.predictor.predict(np.array(history))
                self.obstacles[obstacle_id]['predicted'] = predicted
    
    def get_predicted_positions(self, obstacle_id, step=1):
        """
        Get predicted position for obstacle at future step
        
        Args:
            obstacle_id: Obstacle ID
            step: Prediction step (1, 2, or 3)
        
        Returns:
            position: Predicted (x, y, z) or None if unavailable
        """
        if obstacle_id not in self.obstacles:
            return None
        
        predicted = self.obstacles[obstacle_id].get('predicted')
        if predicted is None or step > len(predicted):
            return None
        
        return predicted[step - 1]
    
    def get_all_predicted_positions(self, step=1):
        """
        Get predicted positions for all obstacles
        
        Args:
            step: Prediction step
        
        Returns:
            positions: Dict {obstacle_id: position}
        """
        positions = {}
        for obs_id in self.obstacles:
            pos = self.get_predicted_positions(obs_id, step)
            if pos is not None:
                positions[obs_id] = pos
        return positions
    
    def should_replan(self, planned_path, current_waypoint_idx, threshold=0.5):
        """
        Determine if replanning is necessary
        
        Args:
            planned_path: Current planned path
            current_waypoint_idx: Current position in path
            threshold: Distance threshold for replanning trigger
        
        Returns:
            should_replan: Boolean
            reason: String describing why replanning is needed
        """
        if current_waypoint_idx >= len(planned_path):
            return False, "Path completed"
        
        # Check upcoming waypoints against predicted obstacles
        look_ahead = min(self.prediction_horizon, 
                        len(planned_path) - current_waypoint_idx)
        
        for i in range(look_ahead):
            waypoint = planned_path[current_waypoint_idx + i]
            step = i + 1
            
            # Check collision with predicted obstacle positions
            predicted_positions = self.get_all_predicted_positions(step)
            
            for obs_id, pred_pos in predicted_positions.items():
                distance = np.linalg.norm(waypoint[:3] - pred_pos)
                
                if distance < threshold:
                    return True, f"Predicted collision with obstacle {obs_id} at step {step}"
        
        return False, "Path is clear"
    
    def visualize_predictions(self):
        """Print current predictions for debugging"""
        print("\n" + "="*60)
        print("Dynamic Obstacle Predictions")
        print("="*60)
        
        for obs_id, data in self.obstacles.items():
            if data['predicted'] is not None:
                current_pos = data['history'][-1][:3]
                print(f"\nObstacle {obs_id}:")
                print(f"  Current: {current_pos}")
                for i, pred in enumerate(data['predicted']):
                    print(f"  Step {i+1}: {pred}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Obstacle Predictor Training/Testing')
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_train', type=int, default=800)
    parser.add_argument('--n_val', type=int, default=200)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Generating training data...")
        train_trajs = generate_training_data(args.n_train, traj_length=50)
        val_trajs = generate_training_data(args.n_val, traj_length=50)
        
        # Save generated data
        with open('training_trajectories.pkl', 'wb') as f:
            pickle.dump({'train': train_trajs, 'val': val_trajs}, f)
        
        # Create datasets
        train_dataset = ObstacleTrajectoryDataset(train_trajs)
        val_dataset = ObstacleTrajectoryDataset(val_trajs)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create and train model
        model = ObstaclePredictorLSTM()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        train_losses, val_losses = train_predictor(
            model, train_loader, val_loader, 
            epochs=args.epochs, device=device
        )
        
        print("\n✓ Training complete!")
        
    else:  # test mode
        print("Testing obstacle predictor...")
        
        # Create predictor
        predictor = ObstaclePredictor('obstacle_predictor_best.pth')
        
        # Test with synthetic trajectory
        print("\nGenerating test trajectory...")
        test_traj = generate_training_data(1, traj_length=30, 
                                          motion_types=['circular'])[0]
        
        history = test_traj[:10, [1,2,3,4,5,6]]  # x,y,z,vx,vy,vz
        ground_truth = test_traj[10:13, 1:4]  # next 3 positions
        
        print(f"\nPredicting from history:")
        print(f"Last position: {history[-1, :3]}")
        print(f"Last velocity: {history[-1, 3:]}")
        
        predictions = predictor.predict(history)
        
        print(f"\nPredictions:")
        for i, pred in enumerate(predictions):
            gt = ground_truth[i]
            error = np.linalg.norm(pred - gt)
            print(f"  Step {i+1}: {pred} (GT: {gt}, Error: {error:.3f})")
        
        # Test dynamic manager
        print("\n\nTesting Dynamic Obstacle Manager...")
        manager = DynamicObstacleManager(predictor)
        
        # Simulate obstacle updates
        for t in range(15):
            pos = test_traj[t, 1:4]
            vel = test_traj[t, 4:7]
            manager.update_obstacle('obs1', pos, vel, t)
            
            if t >= 10:
                pred = manager.get_predicted_positions('obs1', step=1)
                if pred is not None:
                    print(f"t={t}: Predicted next position: {pred}")
        
        print("\n✓ Testing complete!")
