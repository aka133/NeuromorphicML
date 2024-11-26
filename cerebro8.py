import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import List, Dict
import seaborn as sns
import pyNN.utility as utility
import pickle
import gzip
from datasets import load_dataset
from pyNN.parameters import LazyArray, Sequence
from pyNN.space import Space
from pyNN.connectors import DistanceDependentProbabilityConnector
from pyNN.random import RandomDistribution
from openai import OpenAI
import logging
import faiss
from typing import cast, BinaryIO, Optional, Tuple, Union
from scipy.stats import zscore
import os
import gc
from collections import defaultdict
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)

# Determine the simulator backend from command line arguments
import pyNN.neuron as sim


# ==============================================================================
# Parameters
# ==============================================================================

# Simulation Parameters
SIMULATION_TIMESTEP = 2.0  # Increased from 0.5ms to 2.0ms

# Synaptic Weight Parameters
SYNAPSE_WEIGHT_MIN = 5.0
SYNAPSE_WEIGHT_MAX = 15.0

# Neuron Parameters
max_rate = 200.0  # Maximum firing rate in Hz
min_rate = 100.0  # Minimum firing rate in Hz
embedding_dimension = 256  # OpenAI embedding dimension

# OpenAI API Key (Replace with your actual key)
client = OpenAI(api_key=OPENAI_KEY)

# Standard deviation for Gaussian tuning curve
sigma = 0.6

# Embedding cache to store fetched embeddings
embedding_cache = {}

# Excitatory neuron parameters (base values)
exc_neuron_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65.0,
    'd': 8.0,
}

# Inhibitory neuron parameters (base values)
inh_neuron_params = {
    'a': 0.1,
    'b': 0.2,
    'c': -65.0,
    'd': 2.0,
}

# Layer-specific excitatory neuron parameters
layer_exc_params = {
    'Layer_1_Exc': {
        'a': 0.02,
        'b': 0.2,
        'c': -65.0,
        'd': 8.0,  # Regular spiking neurons
    },
    'Layer_2_Exc': {
        'a': 0.02,
        'b': 0.2,
        'c': -55.0,
        'd': 4.0,  # Intrinsically bursting neurons
    },
    'Layer_3_Exc': {
        'a': 0.02,
        'b': 0.2,
        'c': -50.0,
        'd': 2.0,  # Chattering neurons
    },
    'Layer_4_Exc': {
        'a': 0.02,
        'b': 0.2,
        'c': -65.0,
        'd': 8.0,  # Fast spiking neurons
    },
    'Layer_5_Exc': {
        'a': 0.03,
        'b': 0.25,
        'c': -60.0,
        'd': 4.0,  # Thalamo-cortical neurons
    },
    'Layer_6_Exc': {
        'a': 0.1,
        'b': 0.26,
        'c': -60.0,
        'd': 0.0,  # Resonator neurons
    },
}

# Layer-specific inhibitory neuron parameters
layer_inh_params = {
    'Layer_1_Inh': {
        'a': 0.1,
        'b': 0.2,
        'c': -65.0,
        'd': 2.0,  # Fast spiking interneurons
    },
    'Layer_2_Inh': {
        'a': 0.1,
        'b': 0.2,
        'c': -65.0,
        'd': 2.0,
    },
    'Layer_3_Inh': {
        'a': 0.1,
        'b': 0.2,
        'c': -65.0,
        'd': 2.0,
    },
    'Layer_4_Inh': {
        'a': 0.1,
        'b': 0.2,
        'c': -65.0,
        'd': 2.0,
    },
    'Layer_5_Inh': {
        'a': 0.1,
        'b': 0.2,
        'c': -65.0,
        'd': 2.0,
    },
    'Layer_6_Inh': {
        'a': 0.1,
        'b': 0.2,
        'c': -65.0,
        'd': 2.0,
    },
}

# STDP Parameters
STDP_TIMING_PARAMS = {
    'tau_plus': 15.0,
    'tau_minus': 15.0,
    'A_plus': 0.1,
    'A_minus': 0.12
}

STDP_WEIGHT_PARAMS = {
    'w_min': 0.0,
    'w_max': 1000.0
}

# Context Integration Parameters
WORKING_MEMORY_SIZE = 300
TEMPORAL_INTEGRATION_WINDOW = 500.0  # ms
CONTEXT_DIMENSION = 512
CONTEXT_DECAY = 0.9
CONTEXT_UPDATE_RATE = 0.1
REWARD_SEQUENCE_WEIGHT = 70
REWARD_CONTEXT_WEIGHT = 30

class WorkingMemoryCircuit:
    def __init__(self, size=WORKING_MEMORY_SIZE):
        self.population = sim.Population(
            size,
            sim.Izhikevich(
                a=0.002,  # Slower adaptation
                b=0.2,
                c=-65.0,
                d=8.0
            ),
            label='Working_Memory'
        )
        
        # Record with proper temporal resolution
        self.population.record(['spikes', 'v'], sampling_interval=5.0)  # 5ms sampling
        
        # Strong recurrent connections with temporal structure
        self.recurrent_projection = sim.Projection(
            self.population,
            self.population,
            sim.FixedProbabilityConnector(p_connect=0.4),
            synapse_type=sim.StaticSynapse(
                weight=RandomDistribution('normal', mu=500.0, sigma=50.0),
                delay=RandomDistribution('normal', mu=5.0, sigma=1.0)
            ),
            receptor_type='excitatory'
        )

def process_temporal_sequence(spike_counts_sequence, working_memory, index, word_list, embeddings, max_rate, min_rate, word_duration):
    """Process sequence with working memory integration and temporal alignment."""
    context_vector = np.zeros(CONTEXT_DIMENSION)
    predictions = []
    
    integration_window = TEMPORAL_INTEGRATION_WINDOW  # ms
    
    for i, spike_counts in enumerate(spike_counts_sequence):
        try:
            # Compute time windows
            window_end = (i + 1) * word_duration
            window_start = window_end - integration_window
            
            # Get working memory state with error handling
            data = working_memory.population.get_data()
            if len(data.segments) > 0 and len(data.segments[-1].filter(name='v')) > 0:
                wm_activity = data.segments[-1].filter(name='v')[0]
                vm = wm_activity.magnitude
                
                # Get activity within the temporal window
                times = wm_activity.times.rescale('ms').magnitude
                window_mask = (times >= window_start) & (times <= window_end)
                vm_window = vm[window_mask, :]
                
                if len(vm_window) > 0:
                    wm_state = vm_window.mean(axis=0)
                    if wm_state.shape[0] > CONTEXT_DIMENSION:
                        wm_state = wm_state[:CONTEXT_DIMENSION]
                    elif wm_state.shape[0] < CONTEXT_DIMENSION:
                        padding = CONTEXT_DIMENSION - wm_state.shape[0]
                        wm_state = np.pad(wm_state, (0, padding), mode='constant')
                    wm_state = zscore(wm_state)
                else:
                    print(f"Warning: No working memory data in window for element {i}")
                    wm_state = np.zeros(CONTEXT_DIMENSION)
            else:
                print(f"Warning: No working memory data available for element {i}")
                wm_state = np.zeros(CONTEXT_DIMENSION)
            
            # Update context vector with temporal decay and normalization
            context_vector = CONTEXT_DECAY * context_vector + CONTEXT_UPDATE_RATE * wm_state
            
            # Normalize context vector
            context_norm = np.linalg.norm(context_vector)
            if context_norm > 0:
                context_vector = context_vector / context_norm
            
            # Verify spike counts are valid
            spike_counts_sum = np.sum(spike_counts)
            print(f"Spike counts sum for element {i}: {spike_counts_sum}")

            if spike_counts_sum == 0:
                print(f"Warning: No spikes detected for sequence element {i}")
                predictions.append(None)
                continue
            
            # Make prediction using combined information
            prediction = decode_spike_pattern(
                spike_counts,
                context_vector,
                index,
                word_list,
                embeddings,
                max_rate,
                min_rate,
                word_duration,
                k=5  # Number of candidates to consider
            )
            
            # Add prediction with temporal alignment information
            predictions.append({
                'word': prediction,
                'time': window_end,
                'context_strength': np.linalg.norm(context_vector)
            })
            
        except Exception as e:
            print(f"Error processing temporal sequence element {i}: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            predictions.append(None)
            continue
    
    # Post-process predictions to handle None values and temporal consistency
    processed_predictions = []
    for i, pred in enumerate(predictions):
        if pred is not None:
            processed_predictions.append(pred['word'])
        else:
            # Use previous prediction if available, otherwise None
            prev_word = processed_predictions[-1] if processed_predictions else None
            processed_predictions.append(prev_word)
    
    return processed_predictions

def add_variability(layer, params):
    """Add variability to neuron parameters within a layer and initialize 'v' and 'u'."""
    param_values = {}
    # Izhikevich model parameters to vary
    variable_params = ['a', 'b', 'c', 'd']

    for param_name in variable_params:
        if param_name in params:
            base_value = params[param_name]
            # Define variability scales based on biological data
            variability_scales = {
                'a': 0.02,  # Variability for 'a'
                'b': 0.02,  # Variability for 'b'
                'c': 2.0,   # Variability for 'c' in mV
                'd': 2.0    # Variability for 'd'
            }
            scale_value = variability_scales.get(param_name, 0.1)
            if base_value != 0:
                # For parameters with non-zero base values
                values = np.random.normal(
                    loc=base_value,
                    scale=abs(base_value) * scale_value,
                    size=layer.size
                )
            else:
                # For parameters with base value zero
                values = np.random.normal(
                    loc=base_value,
                    scale=scale_value,
                    size=layer.size
                )
            # Clamp values to prevent extreme parameters
            if param_name == 'a':
                values = np.clip(values, 0.001, 0.2)
            elif param_name == 'b':
                values = np.clip(values, 0.005, 1.0)
            elif param_name == 'c':
                values = np.clip(values, -80.0, -50.0)
            elif param_name == 'd':
                values = np.clip(values, 0.0, 20.0)
            param_values[param_name] = values.tolist()

            # Print parameter stats
            print(f"{layer.label} - '{param_name}' parameter: mean={np.mean(values)}, std={np.std(values)}")

    # Set the parameters
    layer.set(**param_values)

    # Initialize 'v' closer to 'c' value for each neuron
    c_values = layer.get('c', gather=False)
    v_initial = np.random.uniform(c_values + 5.0, c_values + 10.0)
    layer.initialize(v=v_initial)

    # Initialize 'u' based on 'b' and 'v_initial'
    b_values = layer.get('b', gather=False)
    u_values = b_values * v_initial
    layer.initialize(u=u_values)

    # Print initialized 'v' and 'u' stats
    print(f"{layer.label} - Initialized 'v': mean={np.mean(v_initial)}, std={np.std(v_initial)}")
    print(f"{layer.label} - Initialized 'u': mean={np.mean(u_values)}, std={np.std(u_values)}\n")

@dataclass
class NetworkMetrics:
    spike_rates: Dict[str, np.ndarray]
    weight_distributions: Dict[str, np.ndarray]
    burst_events: Dict[str, List[float]]
    layer_activities: Dict[str, np.ndarray]
    connection_matrices: Dict[str, np.ndarray]

class NetworkMonitor:
    def __init__(self, layers, projections):
        print("Initializing Network Monitor...")
        self.layers = layers
        self.projections = projections
        self.metrics_history = []
        self.last_timestamp = 0.0  # Initialize last_timestamp
        self.fig_manager = PlotManager()
        self.spike_monitors = {}
        self.state_monitors = {}
        self.recorded_indices = {}

        sampling_interval = 9.75  # Record every 5 ms, adjust as needed
        neurons_to_record = list(range(20))  # List of neuron indices to record from

        for name, layer in layers.items():
            print(f"Setting up recording for {name}...")
            layer.record(['spikes'])
            # Record 'v' from specified neurons without using PopulationView
            layer.record({'v': neurons_to_record}, sampling_interval=sampling_interval)
            self.spike_monitors[name] = layer
            self.state_monitors[name] = layer  # Update the state monitors
            self.recorded_indices[name] = neurons_to_record

        self.connection_projections = {}
        self._collect_projections()
        print("Network Monitor initialized.")

    def _collect_projections(self):
        print("Collecting projections...")
        for proj in self.projections:
            if len(proj) > 0:
                pre_name = proj.pre.label
                post_name = proj.post.label
                proj_name = f"{pre_name}_to_{post_name}"
                self.connection_projections[proj_name] = proj
            else:
                print(f"Projection from {proj.pre.label} to {proj.post.label} has no connections.")
        print("Projections collected.")

    def record_state(self, timestamp: float):
        print(f"Recording network state at time {timestamp} ms...")
        try:
            duration = timestamp - self.last_timestamp  # Duration since last recording

            # Collect metrics
            metrics = NetworkMetrics(
                spike_rates=self._get_spike_rates(duration),
                weight_distributions=self._get_weight_distributions(),
                burst_events=self._detect_bursts(),
                layer_activities=self._get_layer_activities(),
                connection_matrices=self._get_connection_matrices()
            )

            # Append to history if needed (can be limited in size)
            #self.metrics_history.append((timestamp, metrics))
            
            # Update last_timestamp after collecting data
            self.last_timestamp = timestamp
            print("Network state recorded.")

            # After recording, write metrics to disk
            self.save_metrics(timestamp, metrics)

            # Then clear the metrics to free up memory
            self.metrics_history.clear()

            # Return the timestamp and metrics
            return timestamp, metrics

        except Exception as e:
            print(f"An error occurred during network state recording: {e}")
            return None, None

    def save_metrics(self, timestamp, metrics):
        # Implement saving metrics to disk
        filename = f"metrics_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(metrics, f)

    def _get_spike_rates(self, duration: float) -> Dict[str, np.ndarray]:
        """Get spike rates for each layer based on spike times."""
        print("Calculating spike rates...")
        spike_rates = {}
        for name, layer in self.spike_monitors.items():
            print(f"Processing layer: {name}")
            data = layer.get_data()
            if len(data.segments) == 0:
                print(f"No data recorded for {name}.")
                spike_rates[name] = np.zeros(layer.size)
                continue
            spiketrains = data.segments[-1].spiketrains
            del data

            rates = np.zeros(layer.size)
            for idx, st in enumerate(spiketrains):
                # Filter spikes after last_timestamp
                spike_times = st.times.rescale('ms').magnitude
                new_spikes = spike_times[spike_times > self.last_timestamp]
                spike_count = len(new_spikes)
                rates[idx] += spike_count / (duration / 1000.0)  # Hz
            spike_rates[name] = rates
        print("Spike rates calculated.")
        return spike_rates

    def _get_weight_distributions(self) -> Dict[str, np.ndarray]:
        """Get weight distributions for each projection"""
        print("Fetching weight distributions...")
        weight_distributions = {}
        for proj_name, proj in self.connection_projections.items():
            try:
                weights = proj.get('weight', format='array', gather=False)
                weights = weights.ravel()
                weight_distributions[proj_name] = weights
            except Exception as e:
                print(f"Error fetching weights for projection {proj_name}: {e}")
        print("Weight distributions fetched.")
        return weight_distributions

    def _detect_bursts(self) -> Dict[str, List[float]]:
        """Detect burst events in each layer based on spike times."""
        print("Detecting bursts...")
        bursts = {}
        for layer_name, layer in self.spike_monitors.items():
            print(f"Processing layer: {layer_name}")
            data = layer.get_data()
            if len(data.segments) == 0:
                print(f"No data recorded for {layer_name}.")
                bursts[layer_name] = []
                continue
            spiketrains = data.segments[-1].spiketrains

            burst_times = []
            burst_threshold = 3
            time_window = 10.0  # ms

            for idx, st in enumerate(spiketrains):
                spike_times = st.times.rescale('ms').magnitude
                new_spike_times = spike_times[spike_times > self.last_timestamp]
                # Detect bursts in new spikes
                for i in range(len(new_spike_times)):
                    window_spikes = new_spike_times[
                        (new_spike_times >= new_spike_times[i]) &
                        (new_spike_times < new_spike_times[i] + time_window)
                    ]
                    if len(window_spikes) >= burst_threshold:
                        burst_times.append(new_spike_times[i])
            bursts[layer_name] = burst_times
        print("Bursts detected.")
        return bursts

    def _get_layer_activities(self) -> Dict[str, np.ndarray]:
        activities = {}
        window_size = 20.0  # ms

        for name, layer in self.state_monitors.items():
            data = layer.get_data().segments[-1]
            vm = data.filter(name='v')[0]
            times = vm.times.rescale('ms').magnitude

            # Create proper time mask
            current_time = self.last_timestamp
            window_start = max(0, current_time - window_size)
            time_mask = np.logical_and(times >= window_start, times <= current_time)

            # Handle the neo.AnalogSignal properly
            if np.any(time_mask):
                vm_data = vm.magnitude[time_mask]  # Get underlying numpy array
                activities[name] = vm_data
            else:
                print(f"No voltage data in time window for {name}")
                activities[name] = np.array([])

        return activities

    def _get_connection_matrices(self) -> Dict[str, np.ndarray]:
        """Get connection matrices for each projection"""
        print("Fetching connection matrices...")
        connection_matrices = {}
        for proj_name, proj in self.connection_projections.items():
            weights = proj.get('weight', format='array', gather=False)
            connection_matrices[proj_name] = weights
        print("Connection matrices fetched.")
        return connection_matrices
    
    def clear_history(self):
        print("Clearing network monitor history...")
        self.metrics_history.clear()
        # Also clear data from monitors
        for monitor in self.spike_monitors.values():
            monitor.get_data().segments.clear()
        for monitor in self.state_monitors.values():
            monitor.get_data().segments.clear()

class PlotManager:
    def create_summary_plot(self, metrics: NetworkMetrics):
        """Create comprehensive network activity summary"""
        print("Creating summary plot...")

        # Create figure with more explicit spacing
        fig = plt.figure(figsize=(20, 14))

        # Use GridSpec with explicit spacing
        gs = gridspec.GridSpec(3, 2, figure=fig,
                         height_ratios=[1, 1, 1],
                         width_ratios=[1, 1],
                         hspace=0.5,  # Increased vertical spacing
                         wspace=0.4,  # Increased horizontal spacing
                         top=0.95,    # Adjust top margin
                         bottom=0.05,  # Adjust bottom margin
                         left=0.1,    # Adjust left margin
                         right=0.9)   # Adjust right margin

        # Spike Rates (full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_spike_rates(ax1, metrics.spike_rates)
        ax1.set_title('Spike Rates', pad=20)

        # Weight distributions
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_weight_dist(ax2, metrics.weight_distributions)
        ax2.set_title('Synaptic Weight Distributions', pad=20)
        # Move legend outside plot
        ax2.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0.)

        # Layer activities
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_layer_activities(ax3, metrics.layer_activities)
        ax3.set_title('Layer Activities', pad=20)

        # Connection matrix
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_connectivity(ax4, metrics.connection_matrices)
        ax4.set_title('Connectivity Matrix', pad=20)

        # Burst detection
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_bursts(ax5, metrics.burst_events)
        ax5.set_title('Burst Events', pad=20)

        # Don't use tight_layout, we've manually set the spacing
        return fig

    def _plot_spike_rates(self, ax, spike_rates):
        """Plot spike rates for each layer"""
        for layer_name, rates in spike_rates.items():
            ax.plot(rates, label=layer_name)
        ax.set_title('Spike Rates')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Rate (Hz)')
        ax.legend()

    def _plot_weight_dist(self, ax, weight_distributions):
        """Plot weight distributions with variance check"""
        for name, weights in weight_distributions.items():
            if len(weights) > 0:
                variance = np.var(weights)
                if variance > 1e-10:
                    sns.kdeplot(data=weights, ax=ax, label=name)
                else:
                    unique_value = weights[0] if len(weights) > 0 else 0
                    ax.axvline(x=unique_value,
                             label=f"{name} (constant)",
                             linestyle='--',
                             alpha=0.5)

        ax.set_xlabel('Weight')
        ax.set_ylabel('Density')

    def _plot_layer_activities(self, ax, activities):
        """Plot average activity per layer"""
        for layer_name, activity in activities.items():
            if activity.size > 0:
                mean_activity = activity.mean(axis=1)
                ax.plot(mean_activity, label=layer_name)
        ax.set_title('Layer Activities')
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Membrane Potential (mV)')
        ax.legend()

    def _plot_connectivity(self, ax, matrices):
        """Plot network connectivity matrix"""
        for matrix_name, matrix in matrices.items():
            sns.heatmap(matrix, ax=ax, cmap='coolwarm')
            ax.set_title(f'Connectivity: {matrix_name}')
            break  # Just plot one for brevity

    def _plot_bursts(self, ax, burst_events):
        """Plot burst events over time"""
        for layer_name, bursts in burst_events.items():
            ax.scatter(bursts, [layer_name] * len(bursts), alpha=0.3)
        ax.set_title('Burst Events')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Layer')

def precompute_embeddings(tokenizer):
    """
    Precompute embeddings for all tokens in the tokenizer's vocabulary.
    """
    print("Precomputing embeddings using tokenizer vocabulary...")
    embeddings = {}
    tokens = list(tokenizer.get_vocab().keys())

    batch_size = 100  # Adjust based on API limitations
    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch_tokens,
                model="text-embedding-3-large"
            )
            for token, data in zip(batch_tokens, response.data):
                embedding = data.embedding[:embedding_dimension]
                normalized_embedding = normalize_l2(embedding)
                embeddings[token] = normalized_embedding
        except Exception as e:
            print(f"Error fetching embeddings for batch starting at index {i}: {e}")
            continue

    # Save embeddings to a file
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    print("Embeddings precomputed and saved.")
    return embeddings

def build_faiss_index(embeddings):
    """
    Build a Faiss index for efficient nearest neighbor search.
    """
    print("Building Faiss index...")
    word_list = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[word] for word in word_list]).astype('float32')
    
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embedding_matrix)
    print("Faiss index built.")
    return index, word_list

def connect_network(layers, reservoir):
    """Connect cortical layers and integrate the reservoir projecting to Layer 4."""
    print("Connecting network with reservoir projecting to Layer 4...")
    projections = []
    
    # Connect cortical layers
    cortical_projections = connect_layers(layers)
    projections.extend(cortical_projections)
    
    # Connect reservoir to Layer 4 excitatory neurons
    reservoir_to_layer4_proj = connect_reservoir_to_cortex(reservoir, layers)
    projections.extend(reservoir_to_layer4_proj)
    
    # Connect reservoir to cortical layers
    reservoir_projections = connect_reservoir_to_cortex(reservoir, layers)
    projections.extend(reservoir_projections)
    
    print("Network connected with reservoir projecting to Layer 4.")
    return projections

def create_cortical_layers_with_diversity():
    print("Creating cortical layers with neuron diversity...")
    layers = {}

    # Define layer-specific parameters
    layer_properties = {
        'Layer_1': {'num_neurons_exc': 80, 'num_neurons_inh': 20, 'thickness': 0.4},
        'Layer_2': {'num_neurons_exc': 100, 'num_neurons_inh': 20, 'thickness': 0.4},
        'Layer_3': {'num_neurons_exc': 120, 'num_neurons_inh': 30, 'thickness': 0.6},
        'Layer_4': {'num_neurons_exc': 150, 'num_neurons_inh': 30, 'thickness': 0.8},
        'Layer_5': {'num_neurons_exc': 130, 'num_neurons_inh': 30, 'thickness': 1.0},
        'Layer_6': {'num_neurons_exc': 90, 'num_neurons_inh': 20, 'thickness': 1.2},
    }

    # Cortical area dimensions (in mm)
    cortical_area_length = 5.0  # x-dimension
    cortical_area_width = 5.0   # y-dimension

    for i in range(1, 7):
        layer_name = f'Layer_{i}'
        props = layer_properties[layer_name]

        num_exc_neurons = props['num_neurons_exc']
        num_inh_neurons = props['num_neurons_inh']
        total_neurons = num_exc_neurons + num_inh_neurons
        thickness = props['thickness']
        
        cell_params = {'a': [], 'b': [], 'c': [], 'd': []}
        positions = []
        neuron_types = []

        # Get layer-specific neuron parameters
        exc_params_base = layer_exc_params.get(f'{layer_name}_Exc', exc_neuron_params)
        inh_params_base = layer_inh_params.get(f'{layer_name}_Inh', inh_neuron_params)

        # Generate excitatory neurons
        exc_params = generate_neuron_parameters(exc_params_base, num_exc_neurons)
        positions_exc = generate_neuron_positions(
            num_exc_neurons,
            cortical_area_length,
            cortical_area_width,
            layer_depth=thickness
        )
        neuron_types.extend(['exc'] * num_exc_neurons)

        # Gather excitatory parameters and positions
        for param in ['a', 'b', 'c', 'd']:
            cell_params[param].extend(exc_params[param])
        positions.extend(positions_exc)

        # Generate inhibitory neurons
        inh_params = generate_neuron_parameters(inh_params_base, num_inh_neurons)
        positions_inh = generate_neuron_positions(
            num_inh_neurons,
            cortical_area_length,
            cortical_area_width,
            layer_depth=thickness
        )
        neuron_types.extend(['inh'] * num_inh_neurons)

        # Gather inhibitory parameters and positions
        for param in ['a', 'b', 'c', 'd']:
            cell_params[param].extend(inh_params[param])
        positions.extend(positions_inh)

        # Create a single population for the layer
        layer_population = sim.Population(
            total_neurons,
            sim.Izhikevich,
            cellparams=cell_params,
            label=f"{layer_name}"
        )

        # Assign positions to the population
        layer_population.positions = np.array(positions).T

        # Store neuron types for reference during connections
        layer_population.annotate(neuron_types=neuron_types)

        # Store indices of excitatory and inhibitory neurons
        exc_indices = [idx for idx, n_type in enumerate(neuron_types) if n_type == 'exc']
        inh_indices = [idx for idx, n_type in enumerate(neuron_types) if n_type == 'inh']
        layer_population.annotate(exc_indices=exc_indices, inh_indices=inh_indices)

        # Optionally, add variability to neuron parameters
        add_variability(layer_population, exc_params_base if num_exc_neurons > 0 else inh_params_base)

        layers[layer_name] = layer_population

    print("Cortical layers with neuron diversity created.")
    return layers

def generate_neuron_parameters(base_params, num_neurons):
    """Generate neuron parameters with variability for a given neuron type."""
    param_distributions = {}
    for param, base_value in base_params.items():
        if param == 'a':
            # Example variability for 'a'
            param_distributions[param] = np.random.normal(base_value, 0.005, num_neurons)
        elif param == 'b':
            param_distributions[param] = np.random.normal(base_value, 0.02, num_neurons)
        elif param == 'c':
            param_distributions[param] = np.random.normal(base_value, 2.0, num_neurons)
        elif param == 'd':
            param_distributions[param] = np.random.normal(base_value, 2.0, num_neurons)
        else:
            param_distributions[param] = np.full(num_neurons, base_value)
    return param_distributions

def generate_neuron_positions(num_neurons, area_length, area_width, layer_depth):
    """Generate random neuron positions within the specified layer dimensions."""
    x_positions = np.random.uniform(0, area_length, num_neurons)
    y_positions = np.random.uniform(0, area_width, num_neurons)
    z_positions = np.full(num_neurons, layer_depth)
    positions = np.column_stack((x_positions, y_positions, z_positions))
    return positions.tolist()

def create_midbrain_reservoir():
    """Create midbrain-type reservoir neurons."""
    print("Creating midbrain reservoir...")
    reservoir_params = {
        'a': 0.005,
        'b': 0.1,
        'c': -65.0,
        'd': 8.0,
    }
    reservoir_size = 500  # Adjust as needed

    # Assign random positions to reservoir neurons
    positions = np.random.uniform(low=0.0, high=10.0, size=(reservoir_size, 3))
    positions = positions.T

    reservoir = sim.Population(
        reservoir_size,
        sim.Izhikevich,
        cellparams=reservoir_params,
        label='Reservoir'
    )
    reservoir.positions = positions
    print("Midbrain reservoir created.")
    return reservoir

def connect_layers(layers):
    """Connect cortical layers with biologically realistic patterns."""
    print("Connecting cortical layers with biological patterns...")
    projections = []

    # Define delays
    delay = SIMULATION_TIMESTEP

    # Global parameters for connection probabilities
    p_max = 0.9  # Maximum connection probability
    sigma = 1.0  # Controls the decay rate with distance

    # Define random weight distributions
    weight_exc_distr = RandomDistribution('uniform', [SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX])
    weight_inh_distr = RandomDistribution('uniform', [-1, -0.5])

    # Create synapse definitions with random weights
    synapse_exc = sim.StaticSynapse(weight=weight_exc_distr, delay=delay)
    synapse_inh = sim.StaticSynapse(weight=weight_inh_distr, delay=delay)

    # Create a Space object
    space = Space(axes='xyz')

    # Intra-layer connections
    for i in range(1, 7):
        layer_name = f'Layer_{i}'
        layer = layers[layer_name]

        # Retrieve indices from annotations
        exc_indices = layer.annotations['exc_indices']
        inh_indices = layer.annotations['inh_indices']

        # Create subpopulations
        exc_population = sim.PopulationView(layer, exc_indices)
        inh_population = sim.PopulationView(layer, inh_indices)

        # Excitatory to excitatory
        ddp_connector_exc = sim.DistanceDependentProbabilityConnector(
            f"{p_max} * exp(-d**2 / (2 * {sigma}**2))",
            allow_self_connections=False,
        )

        proj_exc_exc = sim.Projection(
            exc_population,
            exc_population,
            connector=ddp_connector_exc,
            synapse_type=synapse_exc,
            receptor_type='excitatory',
            label=f"{layer_name}_Exc_to_Exc"
        )
        projections.append(proj_exc_exc)

        # Excitatory to inhibitory
        proj_exc_inh = sim.Projection(
            exc_population,
            inh_population,
            connector=ddp_connector_exc,
            synapse_type=synapse_exc,
            receptor_type='excitatory',
            label=f"{layer_name}_Exc_to_Inh"
        )
        projections.append(proj_exc_inh)

        # Inhibitory to excitatory
        ddp_connector_inh = sim.DistanceDependentProbabilityConnector(
            f"{p_max} * exp(-d**2 / (2 * {sigma}**2))",
            allow_self_connections=False,
        )

        proj_inh_exc = sim.Projection(
            inh_population,
            exc_population,
            connector=ddp_connector_inh,
            synapse_type=synapse_inh,
            receptor_type='inhibitory',
            label=f"{layer_name}_Inh_to_Exc"
        )
        projections.append(proj_inh_exc)

    print("Cortical layers connected with biological patterns.")
    return projections

def connect_reservoir_to_cortex(reservoir, layers):
    """Connect the reservoir to cortical layers IV and VI."""
    print("Connecting reservoir to cortical layers...")
    projections = []

    # Define delays
    delay = SIMULATION_TIMESTEP

    # Connection probabilities and distance parameters
    p_max = 0.2
    sigma = 1.0

    # Define random weight distribution
    weight_exc_distr = RandomDistribution('uniform', [SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX])

    # Define synapse with random weights
    synapse = sim.StaticSynapse(weight=weight_exc_distr, delay=delay)
    
    space = Space(axes='xyz')

    for layer_name in ['Layer_4', 'Layer_6']:
        if layer_name in layers:
            post_layer = layers[layer_name]

            ddp_connector_exc = sim.DistanceDependentProbabilityConnector(
                f"{p_max} * exp(-d**2 / (2 * {sigma}**2))",
                allow_self_connections=False,
            )

            projection = sim.Projection(
                reservoir,
                post_layer,
                connector=ddp_connector_exc,
                synapse_type=synapse,
                receptor_type='excitatory',
                label=f'Reservoir_to_{layer_name}'
            )
            projections.append(projection)

    print("Reservoir connected to cortical layers.")
    return projections

def connect_cortex_to_readout(layers, readout_layer):
    """Connect cortical layers to the readout layer using STDP."""
    print("Connecting cortical layers to the readout layer...")
    projections = []
    # Connect higher cortical layers to the readout layer
    for layer_name in ['Layer_5', 'Layer_6']:
        if layer_name in layers:
            pre_layer = layers[layer_name]
            connector = sim.FixedProbabilityConnector(p_connect=0.4)  # Increased connectivity

            # Define random initial weights
            initial_weight = RandomDistribution('uniform', [SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX])

            stdp_synapse = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(**STDP_TIMING_PARAMS),
                weight_dependence=sim.AdditiveWeightDependence(w_min=STDP_WEIGHT_PARAMS['w_min'],
                                                               w_max=STDP_WEIGHT_PARAMS['w_max']),
                dendritic_delay_fraction=0.0,
                weight=initial_weight,
                delay=SIMULATION_TIMESTEP
            )

            projection = sim.Projection(
                pre_layer,
                readout_layer,
                connector,
                synapse_type=stdp_synapse,
                receptor_type='excitatory',
                label=f'{layer_name}_to_Readout'
            )
            projections.append(projection)

    print("Cortical layers connected to readout layer.")
    return projections

def connect_reservoir(reservoir):
    """Add recurrent connections within the reservoir using STDP."""
    connector = sim.FixedProbabilityConnector(p_connect=0.4)
    initial_weight = RandomDistribution('uniform', [SYNAPSE_WEIGHT_MIN, SYNAPSE_WEIGHT_MAX])

    stdp_synapse = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(**STDP_TIMING_PARAMS),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=STDP_WEIGHT_PARAMS['w_min'],
            w_max=STDP_WEIGHT_PARAMS['w_max']
        ),
        dendritic_delay_fraction=0.0,
        weight=initial_weight,
        delay=SIMULATION_TIMESTEP
    )
    projection = sim.Projection(
        reservoir,
        reservoir,
        connector,
        synapse_type=stdp_synapse,
        receptor_type='excitatory',
        label='Reservoir_Recurrent'
    )
    return projection

def normalize_l2(x):
    x = np.array(x)
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def get_openai_embedding(word, embeddings):
    """
    Retrieve the precomputed embedding for a word.
    """
    return embeddings.get(word)

def absmean_quantization(W, epsilon=1e-5):
    gamma = np.mean(np.abs(W))
    W_scaled = W / (gamma + epsilon)
    W_quantized = np.clip(np.round(W_scaled), -1, 1)
    return W_quantized, gamma

def quantize_embeddings(embeddings):
    all_embedding_values = np.concatenate([emb for emb in embeddings.values()])
    _, gamma = absmean_quantization(all_embedding_values)
    for word in embeddings:
        embeddings[word], _ = absmean_quantization(embeddings[word], epsilon=1e-5)
    return embeddings, gamma

def encode_input_sequences(
    input_texts,
    embeddings,
    tokenizer,
    start_time=0.0,
    word_duration=100.0
):
    import numpy as np
    from collections import defaultdict

    spike_times_per_neuron = defaultdict(list)
    overall_word_count = 0  # Tracks total words across all texts

    print("Starting optimized encoding of input sequences with one neuron per dimension.")

    for text in input_texts:
        tokens = tokenizer.tokenize(text)
        print(f"Encoding text: '{text}'")
        total_words_so_far = 0  # Reset for each text
        for word in tokens:
            if word in embeddings:
                embedding = embeddings[word]
                word_start_time = start_time + (overall_word_count + total_words_so_far) * word_duration
                word_end_time = word_start_time + word_duration
                
                for dim, value in enumerate(embedding):
                    neuron_idx = dim  # One neuron per dimension

                    if value == 1:
                        rate = max_rate  # High firing rate
                    elif value == -1:
                        rate = min_rate  # Low firing rate
                    else:  # value == 0
                        continue  # Neuron does not fire

                    # Generate Poisson spike times
                    spike_times = generate_poisson_spike_times(
                        rate,
                        word_start_time,
                        word_end_time
                    )
                    spike_times_per_neuron[neuron_idx].extend(spike_times)
            else:
                print(f"Warning: Word '{word}' not in embeddings.")
            total_words_so_far += 1
        overall_word_count += total_words_so_far  # Update overall count after each text

    print("Encoding completed.")

    # Convert spike_times_per_neuron (dict) to a list of lists
    input_population_size = embedding_dimension  # Only one neuron per dimension
    spike_times_list = [[] for _ in range(input_population_size)]
    for neuron_idx, spike_times in spike_times_per_neuron.items():
        spike_times_list[neuron_idx] = spike_times

    return spike_times_list

def create_input_population(spike_times_per_neuron):
    num_inputs = len(spike_times_per_neuron)
    input_population = sim.Population(
        num_inputs,
        sim.SpikeSourceArray(spike_times=spike_times_per_neuron),
        label='Input'
    )
    return input_population

def connect_input_to_network(input_population, layers):
    """Connect input population to Layer 4 excitatory neurons."""
    print("Connecting input to Layer 4 excitatory neurons...")
    input_projections = []

    layer_name = 'Layer_4'
    if layer_name in layers:
        layer = layers[layer_name]
        exc_indices = layer.annotations['exc_indices']  # Get indices of excitatory neurons

        # Build connections from input neurons to excitatory neurons
        connections = []
        for pre_idx in range(input_population.size):
            for post_idx in exc_indices:
                if np.random.rand() < 0.7:  # Connection probability
                    connections.append((pre_idx, post_idx))

        connector = sim.FromListConnector(connections)
        input_projection = sim.Projection(
            input_population,
            layer,
            connector,
            synapse_type=sim.StaticSynapse(weight=50.0, delay=SIMULATION_TIMESTEP),  # Increased weight
            receptor_type='excitatory',
            label='Input_to_Layer_4_Exc'
        )
        input_projections.append(input_projection)
        print(f"Input connected to excitatory neurons of {layer_name}.")

    else:
        print(f"Layer {layer_name} not found in layers.")

    return input_projections

def create_readout_layer(num_neurons, neuron_params=None):
    """Create a spiking readout layer using Izhikevich neurons."""
    if neuron_params is None:
        neuron_params = {
            'a': 0.02,
            'b': 0.2,
            'c': -65.0,
            'd': 10.0,
        }
    readout_layer = sim.Population(
        num_neurons,
        sim.Izhikevich,
        cellparams=neuron_params,
        label='Readout_Layer'
    )
    # Optionally, add variability to neuron parameters
    add_variability(readout_layer, neuron_params)
    return readout_layer

def connect_readout_recurrently(readout_layer):
    """Add recurrent connections within the readout layer."""
    connector = sim.FixedProbabilityConnector(p_connect=0.4)
    stdp_synapse = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=20.0,
            tau_minus=20.0,
            A_plus=0.005,
            A_minus=0.006
        ),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=0.0,
            w_max=1000
        ),
        dendritic_delay_fraction=0.0,
        weight=500.0,  # Initial weight
        delay=SIMULATION_TIMESTEP
    )
    projection = sim.Projection(
        readout_layer,
        readout_layer,
        connector,
        synapse_type=stdp_synapse,
        receptor_type='excitatory',
        label='Readout_Recurrent'
    )
    return projection

def find_nearest_word(reconstructed_embedding, index, word_list):
    """
    Find the nearest word using the Faiss index.
    """
    if reconstructed_embedding is None:
        return None
    reconstructed_embedding = np.array(reconstructed_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(reconstructed_embedding, k=1)
    nearest_idx = indices[0][0]
    nearest_word = word_list[nearest_idx]
    return nearest_word

def decode_spike_pattern(spike_counts, context_vector, index, word_list, embeddings, max_rate, min_rate, word_duration, k=5):
    """Decode spike pattern into a word using the word list, incorporating context."""
    if np.sum(spike_counts) == 0:
        print("Warning: No neural activity detected")
        return None

    # Compute expected spike counts based on encoding rates and word duration
    expected_count_one = max_rate * (word_duration / 1000.0)
    expected_count_minus_one = min_rate * (word_duration / 1000.0)

    # Define thresholds as fractions of expected counts
    high_threshold = expected_count_one * 0.7  # 70% of expected count for '1'
    low_threshold = expected_count_minus_one * 0.7  # 70% of expected count for '-1'
    noise_threshold = 1  # Spike counts below this are considered noise

    quantized_embedding = np.zeros(embedding_dimension)

    for dim in range(embedding_dimension):
        count = spike_counts[dim]
        if count >= high_threshold:
            quantized_embedding[dim] = 1
        elif count >= low_threshold:
            quantized_embedding[dim] = -1
        elif count <= noise_threshold:
            quantized_embedding[dim] = 0
        else:
            # Ambiguous spike count, handle accordingly
            quantized_embedding[dim] = 0

    # Normalize the quantized_embedding
    quantized_embedding = normalize_l2(quantized_embedding)
    quantized_embedding = quantized_embedding.astype('float32').reshape(1, -1)

    # Find the nearest words using the Faiss index
    distances, indices = index.search(quantized_embedding, k=k)
    candidate_indices = indices[0]
    candidate_distances = distances[0]

    # Retrieve candidate words and their embeddings
    candidate_words = [word_list[idx] for idx in candidate_indices]
    candidate_embeddings = np.array([embeddings[word] for word in candidate_words])

    # Compute context similarities
    context_vector_normalized = normalize_l2(context_vector)
    context_vector_normalized = context_vector_normalized.astype('float32').reshape(1, -1)

    # Ensure context vector and embeddings have the same dimension
    if context_vector_normalized.shape[1] != candidate_embeddings.shape[1]:
        # Resize context vector to match the embeddings dimension
        context_vector_resized = context_vector_normalized[0, :embedding_dimension].reshape(1, -1)
    else:
        context_vector_resized = context_vector_normalized

    # Compute cosine similarities
    context_similarities = np.dot(candidate_embeddings, context_vector_resized.T).flatten()

    # Combine Faiss distances and context similarities
    # Since Faiss distances are L2 distances, higher distances mean less similar
    # Convert distances to similarities for combination
    faiss_similarities = -candidate_distances  # Invert distances
    combined_scores = 0.5 * faiss_similarities + 0.5 * context_similarities

    # Select the candidate with the highest combined score
    best_idx = np.argmax(combined_scores)
    predicted_word = candidate_words[best_idx]

    return predicted_word

def simulate_network_batch(spike_times_per_neuron, layers, reservoir, readout_layer, 
                         input_population, input_projections, net_mon, duration, 
                         epoch, batch_num):
    """Optimized simulation with larger chunks and less frequent recording"""
    print(f"Simulating batch {batch_num} in epoch {epoch + 1}")
    
    chunk_size = 50.0  # Increased from 50.0ms to 200.0ms
    total_time = 0.0
    metrics_buffer = []

    while total_time < duration:
        next_chunk = min(chunk_size, duration - total_time)
        sim.run(next_chunk)
        total_time += next_chunk

        #Record metrics less frequently
        if batch_num % 10 == 0 and total_time >= duration * 0.95:  # Only in last 20% of simulation
            timestamp, metrics = net_mon.record_state(sim.get_current_time())
            if metrics is not None:
                metrics_buffer = [(timestamp, metrics)]  # Keep only latest metrics
        
        # Clear data more aggressively
        for layer in layers.values():
            if hasattr(layer, 'get_data'):
                layer.get_data().segments.clear()
        reservoir.get_data().segments.clear()
        readout_layer.get_data().segments.clear()
        gc.collect()

    # Generate plot only for specific batches
    if metrics_buffer and (batch_num % 10 == 0):
        last_timestamp, last_metrics = metrics_buffer[-1]
        summary_fig = net_mon.fig_manager.create_summary_plot(last_metrics)
        summary_fig.savefig(f'summary_plot_epoch{epoch+1}_batch{batch_num}.png')
        plt.close(summary_fig)

    return net_mon

def process_corpus_in_batches(batch_size=4, sequence_length=10, predict_length=5, max_examples=None):
    """Optimized batch processing with larger batch sizes"""
    print("Loading dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train', streaming=True)
    
    batch = []
    example_count = 0
    stride = sequence_length  # Removed overlap to speed up processing
    
    for example in dataset:
        text = example['text'].strip()
        if text:
            words = text.split()
            if len(words) <= sequence_length + predict_length:
                continue
                
            for i in range(0, len(words) - sequence_length - predict_length + 1, stride):
                input_text = ' '.join(words[i:i+sequence_length])
                target_text = ' '.join(words[i+sequence_length:i+sequence_length+predict_length])
                
                batch.append((input_text, target_text))
                example_count += 1
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                    
                if max_examples and example_count >= max_examples:
                    if batch:
                        yield batch
                    return

    if batch:
        yield batch

def apply_reward_modulated_plasticity(
    projection,
    reward,
    learning_rate=0.1,  # Increased learning rate
    w_min=0.0,
    w_max=50.0         # Increased max weight
):
    # Get current weights with addresses
    connections = projection.get('weight', format='list', with_address=True, gather=False)

    # Modify the weights
    new_weights = []
    for pre_idx, post_idx, weight in connections:
        # Ensure weight is a scalar
        if isinstance(weight, (list, np.ndarray)):
            weight = float(weight[0])
        else:
            weight = float(weight)
        # Compute delta_w
        delta_w = reward * learning_rate
        new_weight = weight + delta_w
        # Clip to bounds
        new_weight = max(min(new_weight, w_max), w_min)
        new_weights.append(new_weight)

    # Set updated weights
    projection.set(weight=new_weights)

def add_background_noise(layers):
    print("Adding background noise to layers...")
    noise_weight = 2  # Increased weight
    noise_rate = 20.0  # Increased rate

    for layer in layers.values():
        noise_source = sim.SpikeSourcePoisson(rate=noise_rate)
        noise_population = sim.Population(
            layer.size,
            noise_source,
            label=f"{layer.label}_Noise"
        )
        connector = sim.OneToOneConnector()
        synapse = sim.StaticSynapse(weight=noise_weight, delay=SIMULATION_TIMESTEP)
        sim.Projection(
            noise_population,
            layer,
            connector,
            synapse_type=synapse,
            receptor_type='excitatory',
            label=f"{noise_population.label}_to_{layer.label}"
        )
        print(f"Background noise added to {layer.label}.")

def plot_neuron_potentials(population, recorded_indices, num_neurons=5):
    """
    Plot the membrane potentials of recorded neurons in the given population.

    Parameters:
    - population: The PyNN Population object.
    - recorded_indices: Indices of neurons for which 'v' was recorded.
    - num_neurons: Number of neurons to plot (default is 5).
    """
    # Retrieve the recorded data
    data = population.get_data().segments[-1]
    vm = data.filter(name='v')[0]  # Extract membrane potentials

    # Get the times and convert to milliseconds
    times = vm.times.rescale('ms').magnitude

    plt.figure(figsize=(12, 6))
    num_recorded_neurons = vm.shape[1]

    for idx in range(min(num_neurons, num_recorded_neurons)):
        neuron_idx = idx  # Index in the recorded data
        neuron_id = recorded_indices[neuron_idx]  # Actual neuron ID in the population
        plt.plot(times, vm[:, neuron_idx], label=f'Neuron {neuron_id}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title(f'Membrane Potentials of {population.label}')
    plt.legend()
    plt.show()

def print_projection_info(projection):
    connections = projection.get(['weight', 'delay'], format='list', with_address=True)
    print(f"Projection {projection.label} has {len(connections)} connections.")
    sample_conn = connections[:5]  # Print first 5 connections
    for conn in sample_conn:
        print(f"Pre_idx: {conn[0]}, Post_idx: {conn[1]}, Weight: {conn[2]}, Delay: {conn[3]}")

def main():
    batch_num = 0  # Add batch counter
    num_epochs = 5
    input_time_step = 20.0    # Decreased from 100.0ms to 20.0ms per word

    sim.setup(timestep=SIMULATION_TIMESTEP)

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # You can choose any tokenizer you prefer

    # Load or precompute embeddings
    try:
        with open('embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
            print("Loaded embeddings from file.")
    except FileNotFoundError:
        print("Embeddings file not found. Precomputing embeddings...")
        embeddings = precompute_embeddings(tokenizer)
    
    embeddings, gamma = quantize_embeddings(embeddings)

    # Build Faiss index
    faiss_index, word_list = build_faiss_index(embeddings)

    # Create cortical layers
    layers = create_cortical_layers_with_diversity()
    add_background_noise(layers)

    # Create reservoir
    reservoir = create_midbrain_reservoir()

    # Create readout layer
    readout_layer_size = 1000  # Adjust as needed
    readout_layer = create_readout_layer(num_neurons=readout_layer_size)

    # Create working memory circuit
    working_memory = WorkingMemoryCircuit()

    # Connect the network
    projections = []
    projections += connect_layers(layers)
    projections += connect_reservoir_to_cortex(reservoir, layers)
    projections += connect_cortex_to_readout(layers, readout_layer)
    projections.append(connect_reservoir(reservoir))

    # Connect cortical layers to the readout layer
    cortex_to_readout_projections = connect_cortex_to_readout(layers, readout_layer)
    projections.extend(cortex_to_readout_projections)

    # Recurrent connections within the reservoir
    reservoir_recurrent_proj = connect_reservoir(reservoir)
    projections.append(reservoir_recurrent_proj)

    # Initialize Network Monitor
    net_mon = NetworkMonitor(
        {**layers, 'Reservoir': reservoir, 'Readout_Layer': readout_layer},
        projections
    )

    # Training Loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for batch in process_corpus_in_batches(
            batch_size=4,
            sequence_length=15,
            predict_length=5,
            max_examples=250
        ):
            batch_num += 1  # Increment batch counter
            print(f"Processing batch {batch_num} in epoch {epoch + 1}")
            
            # Process batch
            input_texts, target_texts = zip(*batch)
            
            # Get current simulation time before encoding
            current_sim_time = sim.get_current_time()
            
            # Get spike times with memory management
            spike_times_list = encode_input_sequences(
                input_texts,
                embeddings,
                tokenizer,
                start_time=current_sim_time + 1.0  # Ensure we start after current time
            )

            # Create and connect input population
            input_population = create_input_population(spike_times_list)
            input_projections = connect_input_to_network(input_population, layers)
            
            # Calculate simulation duration based on total words
            total_words = sum(len(text.split()) for text in input_texts) + \
                         sum(len(text.split()) for text in target_texts)
            duration = total_words * input_time_step

            # Simulate network with current batch
            net_mon = simulate_network_batch(
                spike_times_list,
                layers,
                reservoir,
                readout_layer,
                input_population,
                input_projections,
                net_mon,
                duration,
                epoch,
                batch_num
            )

            # Process outputs and apply learning
            for input_index, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
                target_sequence = target_text.split()
                
                # Get spike counts sequence from readout layer
                spike_counts_sequence = get_readout_spike_counts_sequence(
                    readout_layer,
                    sequence_length=len(target_sequence),
                    input_time_step=input_time_step
                )

                # Process with temporal integration
                predictions = process_temporal_sequence(
                    spike_counts_sequence,
                    working_memory,
                    faiss_index,
                    word_list,
                    embeddings,
                    max_rate=max_rate,
                    min_rate=min_rate,
                    word_duration=input_time_step
                )

                # Compute reward and apply learning
                reward = compute_enhanced_reward(
                    predictions,
                    target_sequence,
                    working_memory.population.get_data()
                )

                apply_context_aware_learning(
                    layers,
                    working_memory,
                    projections,
                    reward
                )

            # Cleanup after batch processing
            for proj in input_projections:
                del proj
            del input_projections
            del input_population
            del spike_times_list
            gc.collect()

            # Plot input spikes with word annotations for the first batch
            if epoch == 1 and batch_num == 1:
                plot_input_spikes_with_words(
                    spike_times_list,
                    words=input_texts[0].split(),
                    word_duration=input_time_step,
                    start_time=current_sim_time,
                    num_neurons=50
                )

    print("Training completed.")

    # Proceed with testing using the trained network
    test_text = "Joe is a monkey. He is a "
    predicted_word = test_network(
        test_text,
        readout_layer,
        layers,
        reservoir,
        projections,
        embeddings,       # Pass embeddings
        tokenizer,
        faiss_index,      # Pass the index
        word_list,        # Pass the word list
        input_time_step=input_time_step
    )

    print(f"Network predicted '{predicted_word}' as the next word after '{test_text}'")

    # Plot membrane potentials for diagnostic purposes
    print("Plotting membrane potentials for diagnostic purposes...")
    # Plot neurons from each cortical layer
    for i in range(1, 7):
        layer_name = f'Layer_{i}'
        if layer_name in layers:
            # Get recorded indices from net_mon
            recorded_indices = net_mon.recorded_indices.get(layer_name, [])
            plot_neuron_potentials(layers[layer_name], recorded_indices, num_neurons=5)

    # Plot Reservoir neurons
    recorded_indices_reservoir = net_mon.recorded_indices.get('Reservoir', [])
    plot_neuron_potentials(reservoir, recorded_indices_reservoir, num_neurons=5)

    # Plot Readout Layer neurons
    recorded_indices_readout = net_mon.recorded_indices.get('Readout_Layer', [])
    plot_neuron_potentials(readout_layer, recorded_indices_readout, num_neurons=5)

    # End the simulation
    sim.end()

def test_network(
    test_text,
    readout_layer,
    layers,
    reservoir,
    projections,
    embeddings,
    tokenizer,
    index,
    word_list,
    predict_length=5,
    input_time_step=100.0
):
    print(f"Testing with input: '{test_text}'")

    current_sim_time = sim.get_current_time()
    spike_times_per_neuron = encode_input_sequences(
        [test_text],
        embeddings, 
        tokenizer,
        start_time=current_sim_time
    )

    # Calculate duration for input and predicted sequence
    total_words_in_test = len(test_text.split()) + predict_length
    duration = total_words_in_test * input_time_step

    # Create input population and connect it
    test_input_population = create_input_population(spike_times_per_neuron)
    test_input_projections = connect_input_to_network(test_input_population, layers)
    projections.extend(test_input_projections)

    # Record spikes
    readout_layer.record(['spikes', 'v'])

    # Run the simulation
    print("Running the simulation for testing...")
    sim.run(duration)
    print("Testing simulation completed.")

    # Get the spike counts sequence
    spike_counts_sequence = get_readout_spike_counts_sequence(
        readout_layer,
        sequence_length=predict_length,
        input_time_step=input_time_step
    )

    context_vector = np.zeros(CONTEXT_DIMENSION)

    predicted_sequence = decode_spike_sequence(
        spike_counts_sequence,
        context_vector,
        index,
        word_list,
        max_rate,
        min_rate,
        word_duration=input_time_step
    )

    print(f"Network predicted sequence: {predicted_sequence} after '{test_text}'")

    # Clean up
    for proj in test_input_projections:
        projections.remove(proj)
    test_input_population = None

    return predicted_sequence

def save_network_state(filename, populations, projections, include_input_populations=True):
    """Save the entire network state, including neuron parameters and synaptic connections."""
    network_state = {
        'populations': {},
        'projections': []
    }

    # Save populations
    for label, population in populations.items():
        parameter_names = population.celltype.default_parameters.keys()
        parameters = {}
        for param in parameter_names:
            values = population.get(param, gather=False)
            if isinstance(values, LazyArray):
                if values.is_homogeneous:
                    # Scalar parameter
                    values = values.evaluate(simplify=True)
                    parameters[param] = float(values)
                else:
                    # Per-neuron parameter
                    values = values.evaluate()
                    parameters[param] = values.tolist()
            elif hasattr(values, 'base_value'):  # For quantities with units
                parameters[param] = float(values)
            else:
                parameters[param] = values
        network_state['populations'][label] = {
            'size': population.size,
            'celltype': type(population.celltype).__name__,
            'parameters': parameters
        }

    # Save projections
    for projection in projections:
        try:
            connections = projection.get(['weight', 'delay'], format='list', with_address=True, gather=False)
            projection_data = {
                'pre_label': projection.pre.label,
                'post_label': projection.post.label,
                'connections': connections,
                'receptor_type': projection.receptor_type,
                'label': projection.label
            }

            # Handle synapse type
            if isinstance(projection.synapse_type, sim.STDPMechanism):
                # Retrieve STDP parameters
                timing_dependence = projection.synapse_type.timing_dependence
                weight_dependence = projection.synapse_type.weight_dependence

                # Updated parameter retrieval
                timing_params = extract_parameters(timing_dependence)
                weight_dep_params = extract_parameters(weight_dependence)

                projection_data['stdp_params'] = {
                    'timing_dependence': {
                        'class': type(timing_dependence).__name__,
                        'parameters': timing_params
                    },
                    'weight_dependence': {
                        'class': type(weight_dependence).__name__,
                        'parameters': weight_dep_params
                    }
                }
                projection_data['synapse_type'] = 'stdp_synapse'
            else:
                projection_data['synapse_type'] = 'static_synapse'

            network_state['projections'].append(projection_data)
        except Exception as e:
            print(f"Warning: Failed to save projection {projection.label}: {e}")
            continue

    # Compress and save
    with gzip.open(filename, 'wb') as f:
        pickle.dump(network_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Network state saved successfully.")
    
def compute_enhanced_reward(predictions, target_sequence, working_memory_data):
    """Enhanced reward computation considering working memory state"""
    # Basic sequence matching reward
    sequence_reward = compute_sequence_reward(predictions, target_sequence)
    
    # Add context coherence reward component
    wm_activity = working_memory_data.segments[-1].filter(name='v')[0]
    context_stability = np.std(wm_activity) / np.mean(wm_activity)
    context_reward = np.exp(-context_stability)  # Higher reward for stable context
    
    # Combine rewards
    total_reward = 0.7 * sequence_reward + 0.3 * context_reward
    return total_reward

def get_population_label(population):
    if isinstance(population, sim.PopulationView):
        return population.parent.label
    else:
        return population.label

def apply_context_aware_learning(layers, working_memory, projections, reward):
    """Apply learning with context awareness"""
    
    # Apply to working memory connections
    recurrent_proj = working_memory.recurrent_projection
    if recurrent_proj is not None:
        apply_reward_modulated_plasticity(
            recurrent_proj,
            reward,
            learning_rate=0.1,
            w_min=0.0,
            w_max=50.0
        )
    
    # Collect labels of excitatory layers
    excitatory_layer_labels = [name for name in layers if 'Exc' in name]
    
    # Apply learning to projections connected to excitatory layers
    for projection in projections:
        pre_label = get_population_label(projection.pre)
        post_label = get_population_label(projection.post)
        
        if pre_label in excitatory_layer_labels or post_label in excitatory_layer_labels:
            apply_reward_modulated_plasticity(
                projection,
                reward,
                learning_rate=0.1,
                w_min=0.0,
                w_max=400.0
            )

def extract_parameters(obj):
    params = {}
    # Get all attributes excluding private and methods
    attr_names = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith('_')]
    for name in attr_names:
        value = getattr(obj, name)
        # Convert quantities and arrays to standard types
        if hasattr(value, 'base_value'):  # For quantities with units
            value = float(value)
        elif isinstance(value, LazyArray):
            value = value.evaluate()
            if np.size(value) == 1:
                value = float(value)
            else:
                value = value.tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (float, int, str, bool)):
            pass  # Value is already a basic type
        else:
            # Skip unpickleable or complex types
            continue
        params[name] = value
    return params

def load_network_state(filename):
    """Load the entire network state, including neuron parameters and synaptic connections."""
    with gzip.open(filename, 'rb') as f:
        network_state = pickle.load(f)

    populations = {}
    for label, pop_data in network_state['populations'].items():
        try:
            # Determine cell type
            celltype_class = getattr(sim, pop_data['celltype'])
            # Create population with default parameters
            population = sim.Population(
                pop_data['size'],
                celltype_class(),
                label=label
            )
            # Set parameters per neuron
            parameters = pop_data['parameters']
            for param, values in parameters.items():
                try:
                    if isinstance(values, list) and len(values) == pop_data['size']:
                        # Per-neuron parameter
                        population.set(**{param: Sequence(values)})
                    else:
                        # Scalar parameter
                        population.set(**{param: values})
                except Exception as e:
                    print(f"Warning: Failed to set parameter {param} for population {label}: {e}")
            populations[label] = population
        except Exception as e:
            print(f"Warning: Failed to create population {label}: {str(e)}")
            continue

    # Reconstruct projections
    projections = []
    for proj_data in network_state['projections']:
        try:
            pre_pop = populations[proj_data['pre_label']]
            post_pop = populations[proj_data['post_label']]

            # Create connector from saved connections
            connector = sim.FromListConnector(
                proj_data['connections']
            )

            # Recreate synapse type
            if proj_data['synapse_type'] == 'stdp_synapse':
                # Get timing and weight dependence classes
                timing_dep_class = getattr(sim, proj_data['stdp_params']['timing_dependence']['class'])
                weight_dep_class = getattr(sim, proj_data['stdp_params']['weight_dependence']['class'])

                # Reconstruct timing and weight dependence
                timing_dep_params = proj_data['stdp_params']['timing_dependence']['parameters']
                weight_dep_params = proj_data['stdp_params']['weight_dependence']['parameters']

                timing_dependence = timing_dep_class(**timing_dep_params)
                weight_dependence = weight_dep_class(**weight_dep_params)

                # Recreate STDPMechanism
                synapse_type = sim.STDPMechanism(
                    timing_dependence=timing_dependence,
                    weight_dependence=weight_dependence,
                    dendritic_delay_fraction=0.0,
                    weight=proj_data['connections'][0][2],  # Use the first connection's weight as initial weight
                    delay=proj_data['connections'][0][3]    # Use the first connection's delay as initial delay
                )
            else:
                synapse_type = sim.StaticSynapse()

            # Create projection
            projection = sim.Projection(
                pre_pop,
                post_pop,
                connector,
                synapse_type=synapse_type,
                receptor_type=proj_data['receptor_type'],
                label=proj_data['label']
            )
            projections.append(projection)
        except Exception as e:
            print(f"Warning: Failed to load projection {proj_data['label']}: {str(e)}")
            continue

    print("Network state loaded successfully.")
    return populations, projections

def save_word_patterns(filename, word_to_input_pattern, word_to_output_pattern):
    patterns = {
        'word_to_input_pattern': word_to_input_pattern,
        'word_to_output_pattern': word_to_output_pattern
    }
    with open(filename, 'wb') as f:
        pickle.dump(patterns, f)

def load_word_patterns(filename):
    with open(filename, 'rb') as f:
        patterns = pickle.load(f)
    return patterns['word_to_input_pattern'], patterns['word_to_output_pattern']

def get_readout_spike_counts(readout_layer, input_index, input_time_step):
    """
    Get spike counts from the readout layer for the current input.
    """
    # Retrieve spikes from the readout layer
    spikes = readout_layer.get_data('spikes')
    spike_counts = np.zeros(readout_layer.size)
    spike_trains = spikes.segments[-1].spiketrains

    # Calculate the time window for the current input
    end_time = sim.get_current_time() - (input_index * input_time_step)
    start_time = end_time - input_time_step

    for idx, train in enumerate(spike_trains):
        spike_times = train.times.rescale('ms').magnitude
        # Count spikes within the time window of the current input
        count = np.sum((spike_times >= start_time) & (spike_times < end_time))
        spike_counts[idx] = count

    return spike_counts

def plot_input_spikes_with_words(spike_times_per_neuron, words, word_duration, start_time=0.0, num_neurons=50):
    """
    Plot input spikes with word annotations.

    Parameters:
    - spike_times_per_neuron: List of spike times per neuron.
    - words: List of words in the input text.
    - word_duration: Duration of each word (ms).
    - start_time: Starting time for the spikes (default is 0.0 ms).
    - num_neurons: Number of neurons to plot (default is 50).
    """
    plt.figure(figsize=(12, 6))
    for neuron_idx in range(min(num_neurons, len(spike_times_per_neuron))):
        spike_times = spike_times_per_neuron[neuron_idx]
        plt.vlines(spike_times, neuron_idx - 0.5, neuron_idx + 0.5)

    # Annotate word time windows
    total_words_so_far = 0
    for word_idx, word in enumerate(words):
            word_start_time = start_time + total_words_so_far * word_duration
            word_end_time = word_start_time + word_duration
            plt.axvspan(word_start_time, word_end_time, color='yellow', alpha=0.3)
            plt.text((word_start_time + word_end_time) / 2, num_neurons + 1, word,
                ha='center', va='bottom', rotation=90)
            total_words_so_far += 1

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Input Population Spike Raster with Word Annotations')
    plt.ylim(-1, num_neurons + 5)  # Adjust ylim to accommodate text
    plt.show()

def generate_poisson_spike_times(rate, t_start, t_stop):
    """Generate spike times for a Poisson process within the specified interval."""
    if rate <= 0 or t_stop <= t_start:
        return []

    duration = t_stop - t_start
    expected_spike_count = rate * (duration / 1000.0)

    # Generate at least one spike if expected count is very low
    if expected_spike_count < 1.0:
        num_spikes = 1
    else:
        num_spikes = np.random.poisson(expected_spike_count)

    # Generate spikes uniformly in the interval
    spike_times = np.sort(np.random.uniform(t_start, t_stop, num_spikes))
    return spike_times.tolist()

def get_readout_spike_counts_sequence(readout_layer, sequence_length, input_time_step):
    """
    Get spike counts from the readout layer for each time window in the sequence.
    """
    spike_counts_sequence = []
    current_time = sim.get_current_time()
    for idx in range(sequence_length):
        start_time = current_time - (sequence_length - idx) * input_time_step
        end_time = start_time + input_time_step
        spike_counts = get_spike_counts_for_time_window(readout_layer, start_time, end_time)
        spike_counts_sequence.append(spike_counts)
    return spike_counts_sequence

def get_spike_counts_for_time_window(population, start_time, end_time):
    spikes = population.get_data('spikes')
    spike_counts = np.zeros(population.size)
    spike_trains = spikes.segments[-1].spiketrains

    for idx, train in enumerate(spike_trains):
        spike_times = train.times.rescale('ms').magnitude
        # Count spikes within the time window
        count = np.sum((spike_times >= start_time) & (spike_times < end_time))
        spike_counts[idx] = count

    return spike_counts

def decode_spike_sequence(spike_counts_sequence, context_vector, index, word_list, max_rate, min_rate, word_duration):
    predicted_sequence = []
    for spike_counts in spike_counts_sequence:
        predicted_word = decode_spike_pattern(
            spike_counts,
            context_vector,
            index,
            word_list,
            embeddings=embeddings,
            max_rate,
            min_rate,
            word_duration
        )
        predicted_sequence.append(predicted_word)
    return predicted_sequence

def compute_sequence_reward(predicted_sequence, target_sequence):
    correct = sum(p == t for p, t in zip(predicted_sequence, target_sequence))
    reward = (2 * correct / len(target_sequence)) - 1  # Scaled between -1 and 1
    return reward



if __name__ == "__main__":  
    main()