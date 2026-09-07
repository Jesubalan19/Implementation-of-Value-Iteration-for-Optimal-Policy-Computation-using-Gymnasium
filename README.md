# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
Develop a Python program that applies the Value Iteration algorithm to the FrozenLake-v1 environment provided by Gymnasium. The algorithm should iteratively update the value of each state until convergence and then derive the optimal policy that maximizes the expected cumulative reward.

## Software Requirements
- Python 3.x
- Gymnasium
- NumPy
- Jupyter Notebook / Google Colab / VS Code

## Environment Description
The **FrozenLake-v1** environment is a grid-world problem in which an agent must move from the **Start (S)** state to the **Goal (G)** while avoiding **Holes (H)**.

Grid Used:

```text
F F S F
F H H F
F F G H
F F F H
```

Where:

- **S** – Start State
- **F** – Frozen Surface (Safe)
- **H** – Hole (Terminal State)
- **G** – Goal State (Reward = 1)

The environment is **stochastic (is_slippery=True)**, meaning the intended action may not always be executed.

---

## MDP Representation

An MDP is represented as:

**MDP = (S, A, P, R, γ)**

Where:

- **S** = Set of states (16 states)
- **A** = {Left, Down, Right, Up}
- **P(s'|s,a)** = Transition probability
- **R(s,a,s')** = Reward function
- **γ = 0.99** = Discount factor

---
## Theory

**Value Iteration** is a Dynamic Programming algorithm used to compute the optimal value function of an MDP.

It repeatedly updates the value of each state using the **Bellman Optimality Equation**:

\[
V(s)=\max_a\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V(s')\right]
\]

The iterations continue until the maximum change in the value function is smaller than a predefined threshold.

After convergence, the optimal policy is obtained by selecting the action that gives the highest expected value.

---

## Algorithm

1. Create the FrozenLake environment.
2. Initialize the value function of all states to zero.
3. Repeat until convergence:
   - Compute the value for every possible action.
   - Update each state's value using the Bellman Optimality Equation.
   - Calculate the maximum difference between old and new values.
4. Stop when the difference becomes less than the threshold.
5. Extract the optimal policy by selecting the action with the highest value for every state.
6. Display the optimal value function and policy.
---

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
      "SFFF",
      "FFFF",
      "FHFH",
      "FFFG"
]


env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
# write your code here


# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal value function.
    """

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(num_states)

    iteration = 0

    while True:
        delta = 0
        iteration += 1

        for state in range(num_states):
            v = V[state]
            action_values = []

            for action in range(num_actions):
                q_value = 0

                # Transition probabilities
                for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                    q_value += prob * (reward + gamma * V[next_state])

                action_values.append(q_value)

            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))

        # Check convergence
        if delta < theta:
            break

    # Extract optimal policy
    policy = np.zeros(num_states, dtype=int)

    for state in range(num_states):
        action_values = []

        for action in range(num_actions):
            q_value = 0

            for prob, next_state, reward, done in env.unwrapped.P[state][action]:
                q_value += prob * (reward + gamma * V[next_state])

            action_values.append(q_value)

        policy[state] = np.argmax(action_values)

    return V, policy, iteration

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iteration = value_iteration(env)

print("Optimal Value Function:")
print(V)

print("\nOptimal Policy:")
print(policy)

print("\nNumber of Iterations:", iteration)

# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: Jesubalan A")
print("Register Number: 212223240060")
print("Value Iteration Completed")
print("Number of Iterations:", iteration)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()
```
## Output

<img width="682" height="207" alt="image" src="https://github.com/user-attachments/assets/36533089-9e35-4c71-8d6a-31dedfedbc50" />
<img width="360" height="353" alt="image" src="https://github.com/user-attachments/assets/4891897a-e423-4905-94cc-de959f972fb1" />



## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed after convergence using the Bellman Optimality Equation.

## Inference

From this experiment, it is observed that the Value Iteration algorithm efficiently computes the optimal value of every state by repeatedly applying the Bellman Optimality Equation. Once the value function converges, the optimal policy is extracted by selecting the action with the highest expected return. This demonstrates how Dynamic Programming can solve finite Markov Decision Processes and determine the best sequence of actions for an agent.

