import torch
import torch.nn as nn
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


class LIFNeuron(nn.Module):
    def __init__(self, neuron_type='hidden', tau_mem=20e-3, tau_syn=5e-3, threshold=1.0, dt=1e-3):
        super().__init__()
        self.neuron_type = neuron_type
        self.tau_mem = nn.Parameter(torch.tensor(tau_mem))
        self.tau_syn = nn.Parameter(torch.tensor(tau_syn))
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.dt = dt
        self.reset()

    def reset(self):
        self.membrane_potential = 0
        self.synaptic_current = 0

    def forward(self, input_current):
        self.synaptic_current += (-self.synaptic_current / self.tau_syn + input_current) * self.dt
        self.membrane_potential += (-self.membrane_potential / self.tau_mem + self.synaptic_current) * self.dt
        
        if self.membrane_potential >= self.threshold:
            spike = 1
            self.membrane_potential = 0
        else:
            spike = 0
        
        return spike

class EvolvableSNN(nn.Module):
    def __init__(self, num_neurons, input_size, output_size, connection_prob=0.1, dt=1e-3, max_connections=1000):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.output_size = output_size
        self.dt = dt
        self.max_connections = max_connections
        
        # Create neurons
        self.neurons = nn.ModuleList([
            LIFNeuron('input', dt=dt) for _ in range(input_size)] +
            [LIFNeuron('hidden', dt=dt) for _ in range(num_neurons - input_size - output_size)] +
            [LIFNeuron('output', dt=dt) for _ in range(output_size)]
        )
        
        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(num_neurons, num_neurons))
        self.initialize_weights(connection_prob)
        
        # Initialize dynamic distance matrix
        self.distance_matrix = torch.eye(num_neurons)
        self.update_distances()
        
        # Track current complexity
        self.current_complexity = 1
        
        # Activity tracking for pruning
        self.neuron_activity = torch.zeros(num_neurons)
        self.synapse_activity = torch.zeros_like(self.weights)

    def initialize_weights(self, connection_prob):
        with torch.no_grad():
            mask = torch.rand_like(self.weights) < connection_prob
            self.weights.data = torch.randn_like(self.weights) * 0.1 * mask

    def update_distances(self):
        # Simulate "folding" by randomly updating distances
        num_folds = self.num_neurons // 10  # Update 10% of distances
        for _ in range(num_folds):
            i, j = random.sample(range(self.num_neurons), 2)
            new_distance = torch.rand(1).item()
            self.distance_matrix[i, j] = new_distance
            self.distance_matrix[j, i] = new_distance

    def add_synapse(self):
        if self.count_connections() < self.max_connections:
            i, j = self.select_neurons_for_connection()
            if self.weights[i, j].item() == 0:
                self.weights[i, j] = torch.randn(1) * 0.1
                print(f"Added synapse from neuron {j} to {i}")

    def remove_synapse(self):
        non_zero = torch.nonzero(self.weights)
        if len(non_zero) > 0:
            idx = random.choice(non_zero)
            i, j = idx[0].item(), idx[1].item()
            self.weights[i, j] = 0
            print(f"Removed synapse from neuron {j} to {i}")

    def select_neurons_for_connection(self):
        i = random.randint(0, self.num_neurons - 1)
        distances = self.distance_matrix[i]
        probabilities = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
        j = random.choices(range(self.num_neurons), weights=probabilities)[0]
        return i, j

    def count_connections(self):
        return torch.sum(self.weights != 0).item()

    def forward(self, input_signal):
        time_steps, batch_size, _ = input_signal.shape
        spikes = torch.zeros(time_steps, batch_size, self.num_neurons)
        
        for t in range(time_steps):
            # Process input
            neuron_inputs = torch.matmul(input_signal[t], self.weights[:self.input_size])
            
            # Update neurons
            current_spikes = torch.tensor([neuron(input) for neuron, input in zip(self.neurons, neuron_inputs.t())]).t()
            spikes[t] = current_spikes
            
            # Propagate spikes
            if t < time_steps - 1:
                input_signal[t+1, :, self.input_size:] += torch.matmul(current_spikes, self.weights[self.input_size:])
            
            # Update activity tracking
            self.update_activity(current_spikes)
        
        return spikes

    def update_activity(self, current_spikes):
        self.neuron_activity += current_spikes.sum(dim=0)
        self.synapse_activity += torch.outer(current_spikes.sum(dim=0), current_spikes.sum(dim=0))

    def prune_synapses(self, prune_threshold=0.01):
        normalized_activity = self.synapse_activity / self.synapse_activity.max()
        mask = normalized_activity > prune_threshold
        self.weights.data *= mask
        pruned_count = (~mask).sum().item()
        print(f"Pruned {pruned_count} synapses")
        self.synapse_activity.zero_()

    def prune_neurons(self, prune_threshold=0.01):
        normalized_activity = self.neuron_activity / self.neuron_activity.max()
        neurons_to_prune = (normalized_activity <= prune_threshold).nonzero().squeeze()
        
        # Only prune hidden neurons
        neurons_to_prune = [idx for idx in neurons_to_prune.tolist() 
                            if idx >= self.input_size and idx < self.num_neurons - self.output_size]
        
        if not neurons_to_prune:
            print("No neurons to prune")
            return

        # Sort indices in descending order to avoid index shifting issues
        neurons_to_prune.sort(reverse=True)

        # Remove neurons
        for idx in neurons_to_prune:
            del self.neurons[idx]

        # Update weights
        self.weights = nn.Parameter(self.weights[
            [i for i in range(self.num_neurons) if i not in neurons_to_prune]
        ][:, [i for i in range(self.num_neurons) if i not in neurons_to_prune]])

        # Update distance matrix
        self.distance_matrix = self.distance_matrix[
            [i for i in range(self.num_neurons) if i not in neurons_to_prune]
        ][:, [i for i in range(self.num_neurons) if i not in neurons_to_prune]]

        # Update activity trackers
        self.neuron_activity = self.neuron_activity[
            [i for i in range(self.num_neurons) if i not in neurons_to_prune]
        ]
        self.synapse_activity = self.synapse_activity[
            [i for i in range(self.num_neurons) if i not in neurons_to_prune]
        ][:, [i for i in range(self.num_neurons) if i not in neurons_to_prune]]

        # Update neuron count
        self.num_neurons -= len(neurons_to_prune)

        print(f"Pruned {len(neurons_to_prune)} neurons")
        self.neuron_activity.zero_()

    def mutate_weights(self, mutation_rate=0.1, mutation_scale=0.1):
        with torch.no_grad():
            mask = torch.rand_like(self.weights) < mutation_rate
            self.weights += mask * torch.randn_like(self.weights) * mutation_scale
        print(f"Mutated weights with rate {mutation_rate} and scale {mutation_scale}")

    def mutate_neuron_types(self, mutation_rate=0.05):
        for neuron in self.neurons[self.input_size:-self.output_size]:  # Only mutate hidden neurons
            if random.random() < mutation_rate:
                neuron.neuron_type = random.choice(['hidden', 'inhibitory', 'excitatory'])
        print(f"Mutated neuron types with rate {mutation_rate}")

    def increase_complexity(self):
        self.current_complexity += 1
        # Add more neurons
        new_neurons = [LIFNeuron('hidden', dt=self.dt) for _ in range(5)]  # Add 5 new neurons
        self.neurons[self.input_size:-self.output_size] += new_neurons
        
        # Expand weights matrix
        old_size = self.weights.size(0)
        new_size = old_size + 5
        new_weights = torch.zeros(new_size, new_size)
        new_weights[:old_size, :old_size] = self.weights
        new_weights[old_size:, :old_size] = torch.randn(5, old_size) * 0.1
        new_weights[:old_size, old_size:] = torch.randn(old_size, 5) * 0.1
        self.weights = nn.Parameter(new_weights)
        
        # Update distance matrix
        new_distance_matrix = torch.eye(new_size)
        new_distance_matrix[:old_size, :old_size] = self.distance_matrix
        new_distance_matrix[old_size:, :old_size] = torch.rand(5, old_size)
        new_distance_matrix[:old_size, old_size:] = new_distance_matrix[old_size:, :old_size].t()
        self.distance_matrix = new_distance_matrix
        
        # Update activity tracking tensors
        self.neuron_activity = torch.cat([self.neuron_activity, torch.zeros(5)])
        new_synapse_activity = torch.zeros_like(self.weights)
        new_synapse_activity[:old_size, :old_size] = self.synapse_activity
        self.synapse_activity = new_synapse_activity
        
        self.num_neurons = new_size
        print(f"Increased complexity to level {self.current_complexity}, new neuron count: {self.num_neurons}")

def create_evolvable_snn(num_neurons, input_size, output_size, connection_prob=0.1, dt=1e-3):
    return EvolvableSNN(num_neurons, input_size, output_size, connection_prob, dt)

def evolve_network(initial_network, task, generations=100, population_size=50, mutation_rate=0.1, add_synapse_prob=0.2, complexity_increase_interval=10, prune_interval=5):
    population = [initial_network] + [create_evolvable_snn(initial_network.num_neurons, initial_network.input_size, initial_network.output_size) for _ in range(population_size - 1)]
    best_fitness = float('-inf')
    best_network = None
    fitness_history = []

    for gen in range(generations):
        # Update task complexity periodically
        if gen % complexity_increase_interval == 0 and gen > 0:
            task.increase_complexity()
            for model in population:
                model.increase_complexity()

        # Evaluate fitness
        fitness_scores = []
        for model in population:
            fitness = evaluate_fitness(model, task)
            fitness_scores.append(fitness)

        # Selection
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        survivors = sorted_population[:population_size // 2]

        # Update best network
        if fitness_scores[0] > best_fitness:
            best_fitness = fitness_scores[0]
            best_network = survivors[0]

        fitness_history.append(best_fitness)
        print(f"Generation {gen + 1}, Best Fitness: {best_fitness:.4f}")

        # Periodic pruning
        if gen % prune_interval == 0:
            for model in population:
                model.prune_synapses()
                model.prune_neurons()

        # Create offspring
        offspring = []
        while len(offspring) + len(survivors) < population_size:
            parent = random.choice(survivors)
            child = create_evolvable_snn(parent.num_neurons, parent.input_size, parent.output_size)
            child.load_state_dict(parent.state_dict())
            
            # Mutate
            child.mutate_weights(mutation_rate)
            child.mutate_neuron_types(mutation_rate / 2)
            
            if random.random() < add_synapse_prob:
                child.add_synapse()
            else:
                child.remove_synapse()
            
            child.update_distances()  # Update dynamic distances
            
            offspring.append(child)

        population = survivors + offspring

    return best_network, fitness_history


def stdp_curve(t, A_plus=0.1, A_minus=0.12, tau_plus=20, tau_minus=20):
    """
    Compute the STDP curve.
    t: time difference (post-synaptic spike time - pre-synaptic spike time)
    """
    if t >= 0:
        return A_plus * np.exp(-t / tau_plus)
    else:
        return -A_minus * np.exp(t / tau_minus)

def apply_stdp(model, spike_times, learning_rate=0.01):
    """
    Apply STDP to the network based on spike times.
    """
    for i in range(model.num_neurons):
        for j in range(model.num_neurons):
            if i != j:  # Don't apply STDP to self-connections
                pre_spikes = spike_times[j]
                post_spikes = spike_times[i]
                
                weight_change = 0
                for t_post in post_spikes:
                    for t_pre in pre_spikes:
                        delta_t = t_post - t_pre
                        weight_change += stdp_curve(delta_t)
                
                # Update weight
                with torch.no_grad():
                    model.neuron_weights[i, j] += learning_rate * weight_change
    
    print("Applied STDP to network weights")

def visualize_spike_trains(spike_times, duration, dt):
    """
    Visualize spike trains for all neurons.
    """
    num_neurons = len(spike_times)
    plt.figure(figsize=(12, 6))
    for i, spikes in enumerate(spike_times):
        plt.scatter(spikes, [i] * len(spikes), marker='|', color='black')
    plt.ylim(-1, num_neurons)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.title('Spike Trains')
    plt.show()

def calculate_isi_distribution(spike_times):
    """
    Calculate the Inter-Spike Interval (ISI) distribution.
    """
    all_isis = []
    for neuron_spikes in spike_times:
        if len(neuron_spikes) > 1:
            isis = np.diff(neuron_spikes)
            all_isis.extend(isis)
    
    plt.figure(figsize=(10, 5))
    plt.hist(all_isis, bins=50, edgecolor='black')
    plt.xlabel('Inter-Spike Interval (ms)')
    plt.ylabel('Count')
    plt.title('Inter-Spike Interval Distribution')
    plt.show()

def calculate_spike_train_entropy(spike_times, duration, dt):
    """
    Calculate the entropy of spike trains.
    """
    num_neurons = len(spike_times)
    num_time_bins = int(duration / dt)
    spike_matrix = np.zeros((num_neurons, num_time_bins))
    
    for i, spikes in enumerate(spike_times):
        spike_indices = (np.array(spikes) / dt).astype(int)
        spike_matrix[i, spike_indices] = 1
    
    # Calculate entropy
    p_spike = np.mean(spike_matrix)
    p_no_spike = 1 - p_spike
    entropy = -p_spike * np.log2(p_spike + 1e-10) - p_no_spike * np.log2(p_no_spike + 1e-10)
    
    return entropy

def enhanced_spike_analysis(spike_times, duration, dt):
    """
    Perform enhanced analysis on spike trains.
    """
    visualize_spike_trains(spike_times, duration, dt)
    calculate_isi_distribution(spike_times)
    entropy = calculate_spike_train_entropy(spike_times, duration, dt)
    
    # Calculate overall statistics
    num_neurons = len(spike_times)
    total_spikes = sum(len(spikes) for spikes in spike_times)
    avg_firing_rate = total_spikes / (num_neurons * duration)
    active_neurons = sum(1 for spikes in spike_times if len(spikes) > 0)
    
    summary_stats = {
        "total_spikes": total_spikes,
        "average_firing_rate": avg_firing_rate,
        "active_neurons": active_neurons,
        "silent_neurons": num_neurons - active_neurons,
        "entropy": entropy
    }
    
    print(f"Entropy: {entropy:.4f}")
    print(f"Average Firing Rate: {avg_firing_rate:.2f} Hz")
    print(f"Active Neurons: {active_neurons}/{num_neurons}")
    
    return spike_times, summary_stats

def save_best_network(network, filename):
    """Save the best network to a file."""
    torch.save(network.state_dict(), filename)
    print(f"Network saved to {filename}")

def load_best_network(num_neurons, input_size, output_size, dt, filename):
    """Load the best network from a file."""
    network = create_evolvable_snn(num_neurons, input_size, output_size, dt)
    network.load_state_dict(torch.load(filename))
    print(f"Network loaded from {filename}")
    return network

def visualize_neuron_dynamics(model, input_signal, dt, neuron_indices=None):
    """Visualize the dynamics of selected neurons."""
    model.reset_state()
    time_steps = input_signal.shape[0]
    if neuron_indices is None:
        neuron_indices = range(min(5, model.num_neurons))
    
    membrane_potentials = {i: [] for i in neuron_indices}
    synaptic_currents = {i: [] for i in neuron_indices}
    spikes = {i: [] for i in neuron_indices}
    
    with torch.no_grad():
        for t in range(time_steps):
            output = model(input_signal[t].unsqueeze(0))
            for i in neuron_indices:
                neuron = model.neurons[i]
                membrane_potentials[i].append(neuron.membrane_potential)
                synaptic_currents[i].append(neuron.synaptic_current)
                spikes[i].append(output[0, i].item())
    
    time = np.arange(0, time_steps * dt, dt)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    for i in neuron_indices:
        axes[0].plot(time, membrane_potentials[i], label=f'Neuron {i+1}')
        axes[1].plot(time, synaptic_currents[i], label=f'Neuron {i+1}')
        axes[2].step(time, spikes[i], label=f'Neuron {i+1}')
    
    axes[0].set_ylabel('Membrane Potential')
    axes[1].set_ylabel('Synaptic Current')
    axes[2].set_ylabel('Spikes')
    axes[2].set_xlabel('Time (s)')
    
    for ax in axes:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def summarize_spike_counts(spike_times):
    """Summarize spike counts for each neuron."""
    spike_counts = [len(spikes) for spikes in spike_times]
    total_spikes = sum(spike_counts)
    avg_spikes = total_spikes / len(spike_times)
    max_spikes = max(spike_counts)
    min_spikes = min(spike_counts)
    
    print(f"Total spikes: {total_spikes}")
    print(f"Average spikes per neuron: {avg_spikes:.2f}")
    print(f"Max spikes: {max_spikes}")
    print(f"Min spikes: {min_spikes}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(spike_counts)), spike_counts)
    plt.xlabel('Neuron')
    plt.ylabel('Spike Count')
    plt.title('Spike Counts per Neuron')
    plt.show()

class SineWaveTask:
    def __init__(self, duration=1.0, dt=1e-3, initial_frequency=1.0, max_frequency=10.0):
        self.duration = duration
        self.dt = dt
        self.initial_frequency = initial_frequency
        self.max_frequency = max_frequency
        self.current_frequency = initial_frequency
        self.times = torch.arange(0, self.duration, self.dt)
        self.input_size = 1
        self.complexity_level = 1

    def generate_input_signal(self):
        return torch.sin(2 * np.pi * self.current_frequency * self.times).unsqueeze(1)

    def evaluate(self, output_spikes):
        # Convert spikes to continuous signal
        output_signal = torch.cumsum(output_spikes, dim=0)
        output_signal = output_signal / output_signal.max()  # Normalize

        # Generate target signal
        target_signal = torch.sin(2 * np.pi * self.current_frequency * self.times)

        # Calculate mean squared error
        mse = torch.mean((output_signal - target_signal) ** 2)

        # Convert MSE to fitness (higher is better)
        fitness = 1 / (1 + mse.item())

        return fitness

    def increase_complexity(self):
        self.complexity_level += 1
        self.current_frequency = min(self.current_frequency * 1.5, self.max_frequency)
        print(f"Increased task complexity to level {self.complexity_level}, new frequency: {self.current_frequency:.2f} Hz")

def main_function(initial_neurons=10, task_name="sine_wave"):
    dt = 1e-3
    task = SineWaveTask(dt=dt)
    
    # Create the evolvable SNN
    initial_network = create_evolvable_snn(initial_neurons, task.input_size, task.input_size, dt)
    
    # Evaluate initial network
    initial_fitness, initial_spike_times, _ = evaluate_fitness(initial_network, task)
    print(f"Initial network fitness: {initial_fitness:.4f}")
    
    # Visualize initial network
    visualize_network(initial_network)
    
    # Analyze initial network
    print("Initial Network Analysis:")
    analyze_network(initial_network, task)
    
    # Evolve network
    best_network, fitness_history = evolve_network(
        initial_network, 
        task,
        generations=100, 
        population_size=50, 
        mutation_rate=0.1,
        add_synapse_prob=0.2, 
        complexity_increase_interval=10,
        prune_interval=5
    )

    # Evaluate and visualize evolved network
    final_fitness, final_spike_times, _ = evaluate_fitness(best_network, task)
    print(f"Evolved network fitness: {final_fitness:.4f}")

    visualize_network(best_network)

    # Analyze evolved network
    print("Evolved Network Analysis:")
    analyze_network(best_network, task)

    # Plot fitness history
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title("Fitness History")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

    # Save the best network
    save_best_network(best_network, "best_network.pth")

    # Optionally, load the network to verify
    # loaded_network = load_best_network(best_network.num_neurons, best_network.input_size, best_network.output_size, best_network.dt, "best_network.pth")
    # analyze_network(loaded_network, task)

def simulate_network(model, input_signal, dt):
    time_steps = input_signal.shape[0]
    output_spikes = torch.zeros(time_steps, model.num_neurons)
    
    for t in range(time_steps):
        output_spikes[t] = model(input_signal[t].unsqueeze(0)).squeeze(0)
    
    return output_spikes

def evaluate_fitness(model, task):
    input_signal = task.generate_input_signal()
    output_spikes = simulate_network(model, input_signal, task.dt)
    fitness, spike_times = task.evaluate(output_spikes)
    return fitness, spike_times, input_signal

def analyze_network(model, task):
    # Simulate the network
    output_spikes = simulate_network(model, task.input_signal, task.dt)
    
    # Visualize neuron dynamics
    visualize_neuron_dynamics(model, task.input_signal, task.dt)
    
    # Get spike times
    spike_times = [torch.where(output_spikes[:, i] > 0)[0].tolist() for i in range(output_spikes.shape[1])]
    
    # Summarize spike counts
    summarize_spike_counts(spike_times)
    
    # Perform enhanced spike analysis
    enhanced_spike_analysis(spike_times, task.duration, task.dt)

def visualize_network(model):
    """
    Visualize the network structure
    """
    print(f"Network Structure:")
    print(f"Input size: {model.input_size}")
    print(f"Number of neurons: {model.num_neurons}")
    print(f"Input weight shape: {model.weights[:model.input_size].shape}")
    print(f"Neuron weight shape: {model.weights.shape}")
    print(f"\nConnectivity:")
    print(f"Input connectivity: {(model.weights[:model.input_size] != 0).float().mean().item():.2%}")
    print(f"Neuron connectivity: {(model.weights != 0).float().mean().item():.2%}")
    
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(model.num_neurons):
        G.add_node(f"N{i+1}")
    
    # Add edges (synapses)
    for i in range(model.num_neurons):
        for j in range(model.num_neurons):
            weight = model.weights[i, j].item()
            if abs(weight) > 0.01:
                G.add_edge(f"N{j+1}", f"N{i+1}", weight=weight)
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Spiking Neural Network Structure")
    plt.show()

if __name__ == "__main__":
    main_function()