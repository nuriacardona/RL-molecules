# Imports
from collections import deque
import numpy as np
import random
import torch
# Seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


# REPLAY BUFFER (Prioritizing experiences)
class PrioritizedReplayBuffer:
    def __init__(self, size_buffer, epsilon, alpha, beta, increase):
        """Initialize the replay buffer instance with custom parameters

        Args:
            size_buffer (float): maximum number of experiences that can be stored
                 in the replay buffer.
            epsilon (float): factor to update the priorities of the experiences.
            alpha (float): factor for probability computation.
            beta (float): factor to increase beta.
            increase (float): beta increase.
        """

        # Store the input values of the different variables
        self.size_buffer = size_buffer
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.increase = increase

        # Initialize two deques to store the transitions and their corresponding priorities
        self.stored_experiences = deque(maxlen=self.size_buffer)
        self.priorities = deque(maxlen=self.size_buffer)

    def store_experience(self, state, action, reward, next_state, done, mask):
        """Add transitions to the buffer

        Args:
            state (Data): state at time step t
            action (tuple): action performed to the molecule represented at 'state'
            reward (float): reward obtained by performing 'action'
            next_state (Data): state at time ste t+1 (result of performing 'action' at 'state')
            done (boolean): whether the episode ends or not
            mask (tensor): mask for the q-values
        """

        # Store the input transition
        self.stored_experiences.append((state, action, reward, next_state, done, mask))
        # Initialize the transition's priority (maximum value)
        if self.priorities:
            priority = max(self.priorities)
        else:
            priority = 1.0
        # Store the transition's priority
        self.priorities.append(priority)
    
    def update_priorities(self, indices, td_errors):
        """Update the priorities of the transitions that were sampled in the most recent batch

        Args:
            indices: indices of the transitions that were sampled
            td_errors: td error associated to each transition
        """
	
        for idx, td_error in zip(indices, td_errors.tolist()):
            self.priorities[idx] = abs(td_error) + self.epsilon


    def update_beta(self):
        """Increase beta if less than 1 (then it remains as 1)
        """

        if self.beta < 1.0:
            self.beta = min(1.0, self.beta + self.increase)

    def _get_size(self):
        return len(self.stored_experiences)
    
    def sample_batch_experiences(self, num_experiences):
        """Sample a batch of transitions from the buffer

        Args:
            num_experiences (int): how many experiences will be contained in the sampled batch
        
        Returns:
            batch_experiences (tuple): batch of experiences sampled from the replay buffer
        """

        # Retrieve all current priorities
        all_priorities = np.array(self.priorities)
        # Compute the probability of each experience
        probs = all_priorities**self.alpha / np.sum(all_priorities**self.alpha)
        # Draw the random indices that will be included in the batch
        indices = np.random.choice(self._get_size(), num_experiences, p = probs)
        # Compute all importance weights
        imp_weights = (1 / (self._get_size() * probs)) ** self.beta
        max_weight = np.max(imp_weights)
        imp_weights = imp_weights / max_weight
        # Retrieve the experiences and extract the states, actions, rewards, next states, dones, masks and importance weights
        experiences = [self.stored_experiences[idx] for idx in indices]
        states = [exp[0] for exp in experiences]
        actions = [exp[1] for exp in experiences]
        rewards = [exp[2] for exp in experiences]
        next_states = [exp[3] for exp in experiences]
        dones = [exp[4] for exp in experiences]
        masks = [exp[5] for exp in experiences]
        imp_weights = [imp_weights[idx] for idx in indices]

        # Store the different components of the experiences in the proprer format
        batch_experiences = (
            states,
            list(actions),
            torch.tensor(rewards),
            next_states,
            torch.tensor(dones),
            masks,
            indices,
            torch.tensor(imp_weights)
        )

        return batch_experiences

# Store the Replay Buffer state into a file
def save_replay_buffer(replay_buffer, filename="replay_buffer.pth"):
    buffer_state = {
        "stored_experiences": list(replay_buffer.stored_experiences),  # Convert deque to list
        "priorities": list(replay_buffer.priorities),
        "size_buffer": replay_buffer.size_buffer,
        "alpha": replay_buffer.alpha, 
        "beta": replay_buffer.beta, 
        "increase": replay_buffer.increase, 
        "epsilon": replay_buffer.epsilon
    }
    torch.save(buffer_state, filename)

# Load a Replay Buffer stored in a file
def load_replay_buffer(filename="replay_buffer.pth"):
    buffer_state = torch.load(filename)
    replay_buffer = PrioritizedReplayBuffer(buffer_state["size_buffer"], 
                                            buffer_state["alpha"], 
                                            buffer_state["beta"], 
                                            buffer_state["increase"], 
                                            buffer_state["epsilon"])
    replay_buffer.stored_experiences = deque(buffer_state["stored_experiences"], 
                                            maxlen=buffer_state["size_buffer"])  # Restore deque
    replay_buffer.priorities = deque(buffer_state["priorities"], 
                                    maxlen=buffer_state["size_buffer"])
    return replay_buffer
