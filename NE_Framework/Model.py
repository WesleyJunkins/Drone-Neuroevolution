import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy


class CreateCNNModel(nn.Module):
    """
    CNN model for drone image input.
    Expects flat input of shape (N, 4097): first 4096 = 64x64 grayscale image
    (normalized 0-1), last 1 = speed scalar.
    Conv layers extract spatial features; speed is concatenated after flattening.
    """
    def __init__(self, output_features, verbose=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # -> 16x64x64
        self.pool1 = nn.MaxPool2d(2, 2)                            # -> 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # -> 32x32x32
        self.pool2 = nn.MaxPool2d(2, 2)                            # -> 32x16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64x16x16
        self.pool3 = nn.MaxPool2d(2, 2)                            # -> 64x8x8 = 4096
        # after flatten (4096) + speed scalar (1) = 4097
        self.fc1 = nn.Linear(4097, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_features)
        self.verbose = verbose
        self.output_features = output_features

    def forward(self, x):
        img   = x[:, :4096].view(-1, 1, 64, 64)
        speed = x[:, 4096:4097]

        img = self.pool1(F.relu(self.conv1(img)))
        img = self.pool2(F.relu(self.conv2(img)))
        img = self.pool3(F.relu(self.conv3(img)))
        img = img.view(img.size(0), -1)          # (N, 4096)

        combined = torch.cat([img, speed], dim=1) # (N, 4097)
        out = F.relu(self.fc1(combined))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

    def test(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)

    def save_model(self, save_path, model_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_path = os.path.join(save_path, f"{model_name}.pth")
        torch.save(self.state_dict(), full_path)
        return full_path


class CreateModel(nn.Module):
    """
    MLP model. Expects flat input of shape (N, input_features).
    Kept for reference / ablation; CNN is preferred for image input.
    """
    def __init__(self, input_features, h1, h2, h3, output_features, verbose=False):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, output_features)
        self.verbose = verbose
        self.output_features = output_features

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def test(self, X):
        self.eval()
        with torch.no_grad():
            return self(X)

    def save_model(self, save_path, model_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_path = os.path.join(save_path, f"{model_name}.pth")
        torch.save(self.state_dict(), full_path)
        return full_path


class ModelTrainer:
    """Handles model training with optional mini-batch support."""

    def __init__(self, model, criterion=None, optimizer=None, learning_rate=0.001, verbose=False):
        self.model = model
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        self.verbose = verbose

    def train(self, X, y, epochs=100, batch_size=None):
        """
        Train the model. If batch_size is set, trains in mini-batches
        with shuffled indices each epoch.
        """
        self.model.train()
        n = X.size(0)
        epoch_loss = 0.0

        for epoch in range(epochs):
            if batch_size is None or batch_size >= n:
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss = loss.item()
            else:
                indices = torch.randperm(n)
                total_loss, num_batches = 0.0, 0
                for start in range(0, n, batch_size):
                    idx = indices[start:start + batch_size]
                    outputs = self.model(X[idx])
                    loss = self.criterion(outputs, y[idx])
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                epoch_loss = total_loss / num_batches

            if self.verbose:
                print(f"  Epoch {epoch + 1}/{epochs}  loss={epoch_loss:.6f}")

        return epoch_loss

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

    def evaluate(self, X, y, batch_size=None):
        """Return mean loss over the dataset (batched to save memory, sample-weighted)."""
        self.model.eval()
        n = X.size(0)
        with torch.no_grad():
            if batch_size is None or batch_size >= n:
                return self.criterion(self.model(X), y).item()
            total_weighted, total_samples = 0.0, 0
            for start in range(0, n, batch_size):
                X_b = X[start:start + batch_size]
                y_b = y[start:start + batch_size]
                bs  = X_b.size(0)
                total_weighted += self.criterion(self.model(X_b), y_b).item() * bs
                total_samples  += bs
            return total_weighted / total_samples


class Individual:
    """
    Wrapper around a neural network for the genetic algorithm.
    Supports both 'cnn' (CreateCNNModel) and 'mlp' (CreateModel) architectures.
    """

    def __init__(self, model=None, model_type='cnn',
                 input_features=4097, h1=512, h2=256, h3=128, output_features=3,
                 verbose=False, individual_id=None, device=None):
        if model is None:
            self.device = device if device is not None else torch.device('cpu')
            if model_type == 'cnn':
                self.model = CreateCNNModel(output_features, verbose=verbose).to(self.device)
            else:
                self.model = CreateModel(input_features, h1, h2, h3, output_features, verbose=verbose).to(self.device)
            self.model_type = model_type
            self.architecture = {
                'model_type': model_type,
                'input_features': input_features,
                'h1': h1, 'h2': h2, 'h3': h3,
                'output_features': output_features,
            }
        else:
            self.model = model
            # Infer device from the model's parameters
            self.device = next(model.parameters()).device
            # Infer architecture from the model's layer shapes
            if isinstance(model, CreateCNNModel):
                self.model_type = 'cnn'
                self.architecture = {
                    'model_type': 'cnn',
                    'input_features': 4097,
                    'h1': None, 'h2': None, 'h3': None,
                    'output_features': model.fc3.out_features,
                }
            else:
                self.model_type = 'mlp'
                self.architecture = {
                    'model_type': 'mlp',
                    'input_features': model.fc1.in_features,
                    'h1': model.fc1.out_features,
                    'h2': model.fc2.out_features,
                    'h3': model.fc3.out_features,
                    'output_features': model.fc4.out_features,
                }

        self.fitness = None
        self.individual_id = individual_id
        self.generation = 0
        self.parent_ids = []
        self.metadata = {
            'evaluated': False,
            'survived_generations': 0,
            'mutations_applied': 0,
            'crossover_count': 0,
        }

    def _new_model(self):
        """Create a fresh model with the same architecture on the same device."""
        arch = self.architecture
        if arch['model_type'] == 'cnn':
            return CreateCNNModel(arch['output_features'], verbose=self.model.verbose).to(self.device)
        else:
            return CreateModel(arch['input_features'], arch['h1'], arch['h2'],
                               arch['h3'], arch['output_features'], verbose=self.model.verbose).to(self.device)

    def evaluate_fitness(self, X, y, loss_fn=None, batch_size=None):
        """
        Compute fitness = negative MSE loss (higher is better).
        batch_size controls evaluation memory usage.
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        n = X.size(0)
        self.model.eval()
        with torch.no_grad():
            if batch_size is None or batch_size >= n:
                loss_value = loss_fn(self.model(X), y).item()
            else:
                # Weight each batch by its sample count so the grand mean is exact
                total_weighted, total_samples = 0.0, 0
                for start in range(0, n, batch_size):
                    X_b = X[start:start + batch_size]
                    y_b = y[start:start + batch_size]
                    bs  = X_b.size(0)
                    total_weighted += loss_fn(self.model(X_b), y_b).item() * bs
                    total_samples  += bs
                loss_value = total_weighted / total_samples
        self.fitness = -loss_value
        self.metadata['evaluated'] = True
        return self.fitness

    def clone(self, new_id=None):
        cloned_model = self._new_model()
        cloned_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        cloned = Individual(model=cloned_model, individual_id=new_id)
        cloned.metadata = copy.deepcopy(self.metadata)
        cloned.metadata['evaluated'] = False
        cloned.generation = self.generation
        cloned.parent_ids = [self.individual_id] if self.individual_id else []
        return cloned

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.rand_like(param) < mutation_rate
                param[mask] += (torch.randn_like(param) * mutation_strength)[mask]
        self.metadata['mutations_applied'] += 1
        self.metadata['evaluated'] = False

    def crossover(self, other, crossover_rate=0.5):
        offspring_model = self._new_model()
        with torch.no_grad():
            for ps, po, poff in zip(self.model.parameters(),
                                    other.model.parameters(),
                                    offspring_model.parameters()):
                mask = torch.rand_like(ps) < crossover_rate
                poff.data = ps.data.clone()
                poff.data[mask] = po.data[mask]
        offspring = Individual(model=offspring_model)
        offspring.generation = max(self.generation, other.generation) + 1
        offspring.parent_ids = [self.individual_id, other.individual_id]
        offspring.metadata['crossover_count'] = 1
        self.metadata['crossover_count'] += 1
        other.metadata['crossover_count'] += 1
        return offspring

    def save(self, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not file_path.endswith('.pth'):
            file_path = file_path + '.pth'
        torch.save(self.model.state_dict(), file_path)
        return file_path

    def load(self, model_path):
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=self.device))
        self.metadata['evaluated'] = False

    def test(self, X):
        return self.model.test(X)

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        fitness_str = f"fitness={self.fitness:.6f}" if self.fitness is not None else "fitness=None"
        id_str = f"id={self.individual_id}" if self.individual_id else "id=None"
        return f"Individual({id_str}, {fitness_str}, gen={self.generation}, type={self.model_type})"

    def __lt__(self, other):
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness > other.fitness  # higher fitness = better

    def __hash__(self):
        return hash(self.individual_id)

    def __eq__(self, other):
        return self.individual_id == other.individual_id if isinstance(other, Individual) else False
