def generate_grid_world(rows, cols, start_state, goal_state):
    # Define the grid world (states)
    states = [(i, j) for i in range(rows) for j in range(cols)]

    # Define possible actions (up, down, left, right)
    actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    # Define the state transition function
    def transition(state, action):
        new_state = (state[0] + action[0], state[1] + action[1])
        if new_state in states:
            return new_state
        return state  # Stay in the same state if the action leads to an invalid state

    # Define the rewards for each state (assuming all states have a reward of -1 except the goal state)
    rewards = {(i, j): -1 for i in range(rows) for j in range(cols)}
    rewards[goal_state] = 1  # The goal state with a reward of 1

    # Define the discount factor
    gamma = 0.9

    # Define a policy (agent's strategy) - deterministic for simplicity
    policy = {
        (0, 0): 'R',  # Move right when in (0, 0)
        (0, 1): 'R',
        (0, 2): 'U',
        (1, 0): 'R',
        (1, 2): 'U',
        (2, 0): 'R',
        (2, 1): 'R',
        (2, 2): 'U',  # Move up when in (2, 2)
    }

    # Perform value iteration to find the optimal values of each state
    V = {state: 0 for state in states}

    while True:
        delta = 0
        for state in states:
            if state not in policy:
                continue
            v = V[state]
            action = policy[state]
            next_state = transition(state, actions[action])
            reward = rewards[state]
            V[state] = reward + gamma * V[next_state]
            delta = max(delta, abs(v - V[state]))
        if delta < 1e-6:
            break

    # Return the states, actions, rewards, gamma, policy, and values
    return states, actions, rewards, gamma, policy, V

# Example usage
rows, cols = 3, 3
start_state = (0, 0)
goal_state = (2, 2)
states, actions, rewards, gamma, policy, V = generate_grid_world(rows, cols, start_state, goal_state)

# Print the values of each state
for i in range(rows):
    for j in range(cols):
        state = (i, j)
        print(f"State {state}: Value = {V[state]:.2f}")
