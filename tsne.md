## Algorithm: t-Distributed Stochastic Neighbor Embedding (t-SNE)

**Purpose**: t-SNE is a non-linear dimensionality reduction algorithm. It takes high-dimensional data ($X$) and embeds it into a low-dimensional map ($Z$), typically 2D or 3D, while preserving the local neighborhood structure. It achieves this by converting pairwise distances into probabilities and minimizing the mismatch between the high-dimensional distribution and the low-dimensional distribution.

---

### Formula 1: High-Dimensional Similarities ($P_{ij}$)

This formula converts the Euclidean distances between high-dimensional points into conditional probabilities that represent similarity.

For points $x_i$ and $x_j$:

$$
P_{j|i} = \frac{ \exp( -\|x_i - x_j\|^2 / 2\sigma_i^2 ) }{ \sum_{k \neq i} \exp( -\|x_i - x_k\|^2 / 2\sigma_i^2 ) }
$$

To make the matrix symmetric (since similarity should be mutual), we symmetrize it:

$$
P_{ij} = \frac{P_{j|i} + P_{i|j}}{2N}
$$

*(Note: In the simplified example, we used $\sigma_i = 1$ for all points and directly computed $P_{ij}$ over all pairs to sum to 1).*

**How it is used**: 
- Compute all pairwise Euclidean distances in the original high-dimensional space.
- $\sigma_i$ (the bandwidth of the Gaussian kernel) is determined by a hyperparameter called **perplexity** (roughly, the effective number of neighbors).
- This produces a fixed **target matrix $P$** which represents the "ground truth" similarities. **This matrix never changes during training.**

---

### Formula 2: Low-Dimensional Similarities ($Q_{ij}$)

This formula converts the Euclidean distances between the current low-dimensional map points into probabilities using a **Student-t distribution** (which has heavy tails).

$$
Q_{ij} = \frac{ (1 + \|z_i - z_j\|^2)^{-1} }{ \sum_{k \neq l} (1 + \|z_k - z_l\|^2)^{-1} }
$$

**How it is used**:
- At every single training step, take the current coordinates $Z$ (your AI's current guess for the 2D positions).
- Compute all pairwise distances in this low-dimensional space.
- Plug them into this formula to get matrix $Q$.
- Since $Z$ changes every iteration, **$Q$ is updated at every single training step**.

---

### Formula 3: The Loss Function (Kullback-Leibler Divergence)

This is the objective function—the "score" that tells the AI how well its current low-dimensional map matches the original high-dimensional structure.

$$
\boxed{
\mathcal{L} = \sum_{i \neq j} P_{ij} \cdot \log \left( \frac{P_{ij}}{Q_{ij}} \right)
}
$$

**How it is used**:
- Plug the fixed matrix $P$ (from Formula 1) and the current matrix $Q$ (from Formula 2) into this equation.
- The output $\mathcal{L}$ is a single floating-point number.
- **The AI's goal is to minimize this number.** 
  - If $\mathcal{L} = 0$, then $P = Q$, meaning the low-dimensional map perfectly preserves the high-dimensional neighborhood structure.
  - Because the Student-t distribution has heavy tails, this loss heavily penalizes nearby points in high-D that are far apart in low-D (false negatives), but tolerates far points in high-D that are nearby in low-D.

---

### Formula 4: The Gradient (The Learning Signal)

To minimize the loss, the AI must move the low-dimensional points $Z$. The gradient tells us *in which direction* to move each point $z_i$ to reduce the loss.

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial z_i} = 4 \sum_{j \neq i} (P_{ij} - Q_{ij}) \cdot (z_i - z_j) \cdot (1 + \|z_i - z_j\|^2)^{-1}
}
$$

**How it is used**:
- Compute this derivative for every single low-dimensional point $z_i$ (where $i = 1, 2, ..., N$).
- The result is a **vector** (e.g., a 2D or 3D vector) that points in the direction of steepest ascent of the loss.
- **Interpretation of the terms**:
  - $(P_{ij} - Q_{ij})$: If **positive**, the points are **attracted** together (pulled closer). If **negative**, they are **repelled** (pushed apart).
  - $(z_i - z_j)$: The direction of the spring force.
  - $(1 + \|z_i - z_j\|^2)^{-1}$: The heavy-tail weighting. This ensures that repulsion never drops to zero, even when points are very far apart, preventing the map from collapsing into a single clump.

---

### Formula 5: The Optimizer Update Rule (Stochastic Gradient Descent)

Once we have the gradient, we use it to update the positions of the points.

$$
\boxed{
z_i^{new} = z_i^{old} - \eta \cdot \frac{\partial \mathcal{L}}{\partial z_i}
}
$$

Where:
- $\eta$ is the **learning rate** (a hyperparameter, e.g., $0.1$ or $10$ depending on the dataset).
- $\frac{\partial \mathcal{L}}{\partial z_i}$ is the gradient calculated in Formula 4.

**How it is used**:
- For every point $z_i$, subtract $\eta \times \text{its gradient}$ from its current coordinates.
- Because we subtract the gradient, we are moving the points *downhill* in the loss landscape, actively reducing $\mathcal{L}$.

---

### The Complete Training Pipeline (How They All Chain Together)

Here is exactly how an AI uses these 5 formulas to learn the perfect 2D map:

1.  **Initialize**: Start with the high-dimensional data $X$. Compute the fixed target matrix $P$ **once** (Formula 1). Then, initialize $Z$ randomly (e.g., using random Gaussian noise).
2.  **Loop (for thousands of iterations)**:
    - **Step A**: Use the current $Z$ to compute the current low-D similarities $Q$ (Formula 2).
    - **Step B**: Compute the current Loss $\mathcal{L}$ to measure how bad the current map is (Formula 3). (This is just for logging; we don't use it for updates).
    - **Step C**: Compute the Gradient for every point $z_i$ (Formula 4). This tells the AI which points are too close together and which are too far apart.
    - **Step D**: Apply the Optimizer Update to move every point $z_i$ slightly in the right direction (Formula 5).
3.  **Stop**: Repeat the loop until the loss stops decreasing (convergence). The final $Z$ is the beautiful 2D visualization of your high-dimensional data.

---

### Summary Table of the Formulas

| Name | Formula | When is it calculated? | What does it do? |
| :--- | :--- | :--- | :--- |
| **High-D Similarities** | $P_{ij} = \frac{ \exp(-\|x_i - x_j\|^2 / 2\sigma_i^2) }{ \sum_{k \neq l} \exp(-\|x_k - x_l\|^2 / 2\sigma_i^2) }$ | **Once** (before training) | Establishes the target "ground truth" relationships. |
| **Low-D Similarities** | $Q_{ij} = \frac{ (1 + \|z_i - z_j\|^2)^{-1} }{ \sum_{k \neq l} (1 + \|z_k - z_l\|^2)^{-1} }$ | **Every iteration** | Measures the current relationships in the AI's map. |
| **Loss Function** | $\mathcal{L} = \sum_{i \neq j} P_{ij} \log( \frac{P_{ij}}{Q_{ij}} )$ | **Every iteration** | Quantifies how badly the AI's map is doing. |
| **Gradient** | $\frac{\partial \mathcal{L}}{\partial z_i} = 4 \sum_{j \neq i} (P_{ij} - Q_{ij})(z_i - z_j)(1 + \|z_i - z_j\|^2)^{-1}$ | **Every iteration** | Tells the AI exactly *which direction* to move each point. |
| **Update Rule (SGD)** | $z_i^{new} = z_i^{old} - \eta \cdot \frac{\partial \mathcal{L}}{\partial z_i}$ | **Every iteration** | Physically moves the points to reduce the loss. |

---

# t-SNE from Scratch: A Complete Walkthrough with 3 Data Points Example 

This repository walks through a **complete, start-to-finish** example of t-SNE (t-Distributed Stochastic Neighbor Embedding) using exactly 3 data points.

We will compute:
- High-Dimensional similarities ($P$)
- Low-Dimensional similarities ($Q$)
- The Total Loss ($\mathcal{L}$)
- The **full gradients** for every point
- One full optimization step using Stochastic Gradient Descent (SGD)

---

## The Setup

- **High-Dimensional Data ($X$)**: 3 points in 2D space.
  - $x_1 = (0, 0)$
  - $x_2 = (1, 0)$
  - $x_3 = (0, 1)$

- **Low-Dimensional Map ($Z$)**: Our AI's initial random guess for where to put these points in 2D space.
  - $z_1 = (0.0, 0.0)$
  - $z_2 = (0.5, 0.0)$
  - $z_3 = (0.0, 0.5)$

- **Hyperparameters**: 
  - $\sigma = 1$ (simplified for all points)
  - Learning Rate $\eta = 0.1$

---

## Step 1: Calculate High-D Similarities ($P_{ij}$)

We use the formula:

$$
P_{ij} = \frac{ \exp(-\|x_i - x_j\|^2 / 2) }{ \sum_{k<l} \exp(-\|x_k - x_l\|^2 / 2) }
$$

*(Note: We only calculate for $i < j$ so the probabilities sum to 1).*

### Squared Distances in High-D:

- $\|x_1 - x_2\|^2 = (0-1)^2 + (0-0)^2 = \mathbf{1}$
- $\|x_1 - x_3\|^2 = (0-0)^2 + (0-1)^2 = \mathbf{1}$
- $\|x_2 - x_3\|^2 = (1-0)^2 + (0-1)^2 = 1 + 1 = \mathbf{2}$

### Numerators ($\exp(-d^2 / 2)$):

- For $d=1$: $e^{-1/2} = e^{-0.5} \approx 0.6065$
- For $d=1$: $e^{-1/2} = 0.6065$
- For $d=2$: $e^{-2/2} = e^{-1} \approx 0.3679$

### Denominator (Total Sum):

$$
0.6065 + 0.6065 + 0.3679 = 1.5809
$$

### Final $P$ Matrix (Symmetric, sums to 1):

$$
\boxed{
P_{12} = \frac{0.6065}{1.5809} = 0.3836, \quad
P_{13} = 0.3836, \quad
P_{23} = \frac{0.3679}{1.5809} = 0.2328
}
$$

---

## Step 2: Calculate Low-D Map Similarities ($Q_{ij}$)

We use the t-SNE formula with a Student-t distribution:

$$
Q_{ij} = \frac{ (1 + \|z_i - z_j\|^2)^{-1} }{ \sum_{k<l} (1 + \|z_k - z_l\|^2)^{-1} }
$$

### Squared Distances in Low-D:

- $\|z_1 - z_2\|^2 = (0-0.5)^2 + (0-0)^2 = 0.25$
- $\|z_1 - z_3\|^2 = (0-0)^2 + (0-0.5)^2 = 0.25$
- $\|z_2 - z_3\|^2 = (0.5-0)^2 + (0-0.5)^2 = 0.25 + 0.25 = 0.5$

### Numerators ($(1 + d^2)^{-1}$):

- For $d^2=0.25$: $\frac{1}{1.25} = 0.8$
- For $d^2=0.25$: $\frac{1}{1.25} = 0.8$
- For $d^2=0.5$: $\frac{1}{1.5} \approx 0.6667$

### Denominator (Total Sum):

$$
0.8 + 0.8 + 0.6667 = 2.2667
$$

### Final $Q$ Matrix (Symmetric, sums to 1):

$$
\boxed{
Q_{12} = \frac{0.8}{2.2667} = 0.3529, \quad
Q_{13} = 0.3529, \quad
Q_{23} = \frac{0.6667}{2.2667} = 0.2941
}
$$

---

## Step 3: Calculate the Total Loss ($\mathcal{L}$)

We use the KL Divergence:

$$
\mathcal{L} = \sum_{i<j} P_{ij} \log\left( \frac{P_{ij}}{Q_{ij}} \right)
$$

### Per-Pair Calculations:

- **Pair (1,2)**: 
  $0.3836 \times \log(0.3836 / 0.3529) = 0.3836 \times \log(1.0870) = 0.3836 \times 0.0834 = \mathbf{0.0320}$

- **Pair (1,3)**: 
  $0.3836 \times \log(0.3836 / 0.3529) = \mathbf{0.0320}$

- **Pair (2,3)**: 
  $0.2328 \times \log(0.2328 / 0.2941) = 0.2328 \times \log(0.7915) = 0.2328 \times (-0.2337) = \mathbf{-0.0544}$

### Total Loss:

$$
\boxed{\mathcal{L} = 0.0320 + 0.0320 - 0.0544 = 0.0096}
$$

*(It's small, but greater than 0, meaning our map isn't perfect yet).*

---

## Step 4: The Magic Part – Calculate the Gradient for EVERY Point

We use the t-SNE gradient formula:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = 4 \sum_{j \neq i} (P_{ij} - Q_{ij}) \cdot (z_i - z_j) \cdot (1 + \|z_i - z_j\|^2)^{-1}
$$

### Gradient for $z_1$ (starts at $[0.0, 0.0]$)

- **From $j=2$**: 
  $4 \times (0.3836 - 0.3529) \times (z_1 - z_2) \times 0.8$
  $= 4 \times (0.0307) \times (-0.5, 0) \times 0.8 = (-0.0491, 0)$

- **From $j=3$**: 
  $4 \times (0.3836 - 0.3529) \times (z_1 - z_3) \times 0.8$
  $= 4 \times (0.0307) \times (0, -0.5) \times 0.8 = (0, -0.0491)$

- **Total Gradient for $z_1$**: 
  $$
  \boxed{(-0.0491, -0.0491)}
  $$

---

### Gradient for $z_2$ (starts at $[0.5, 0.0]$)

- **From $j=1$**: 
  $4 \times (0.3836 - 0.3529) \times (z_2 - z_1) \times 0.8$
  $= 4 \times (0.0307) \times (0.5, 0) \times 0.8 = (0.0491, 0)$

- **From $j=3$**: 
  $4 \times (0.2328 - 0.2941) \times (z_2 - z_3) \times 0.6667$
  $= 4 \times (-0.0613) \times (0.5, -0.5) \times 0.6667$
  $= (-0.0816, 0.0816)$

- **Total Gradient for $z_2$**: 
  $$
  \boxed{(-0.0325, 0.0816)}
  $$

---

### Gradient for $z_3$ (starts at $[0.0, 0.5]$)

- **From $j=1$**: 
  $4 \times (0.3836 - 0.3529) \times (z_3 - z_1) \times 0.8$
  $= 4 \times (0.0307) \times (0, 0.5) \times 0.8 = (0, 0.0491)$

- **From $j=2$**: 
  $4 \times (0.2328 - 0.2941) \times (z_3 - z_2) \times 0.6667$
  $= 4 \times (-0.0613) \times (-0.5, 0.5) \times 0.6667$
  $= (0.0816, -0.0816)$

- **Total Gradient for $z_3$**: 
  $$
  \boxed{(0.0816, -0.0325)}
  $$

---

## Step 5: The Optimizer Update (SGD)

The update rule is:

$$
z_i^{new} = z_i^{old} - \eta \cdot \text{Gradient}
$$

We use $\eta = 0.1$.

### Update $z_1$:

$$
(0.0, 0.0) - 0.1 \times (-0.0491, -0.0491) = \boxed{(0.0049, 0.0049)}
$$

### Update $z_2$:

$$
(0.5, 0.0) - 0.1 \times (-0.0325, 0.0816) = (0.5, 0) + (0.00325, -0.00816) = \boxed{(0.50325, -0.00816)}
$$

### Update $z_3$:

$$
(0.0, 0.5) - 0.1 \times (0.0816, -0.0325) = (0, 0.5) + (-0.00816, 0.00325) = \boxed{(-0.00816, 0.50325)}
$$

---

## Step 6: What Just Happened? (The Physics Intuition)

Look at how the points moved:

1. **Points 1 and 2**: 
   - In high-D, they had $P=0.3836$ (very similar).
   - In low-D, they had $Q=0.3529$ (slightly less similar).
   - The AI realized they were **too far apart**.
   - The gradient pushed $z_1$ **right** (towards $z_2$) and pushed $z_2$ slightly **right** as well, shrinking the gap from $0.5$ to $\approx 0.498$.
   - They were **attracted** together.

2. **Points 2 and 3**: 
   - In high-D, they were fairly different ($P=0.2328$).
   - In low-D, they were quite close ($Q=0.2941$).
   - The AI realized they were **too close together**.
   - The gradient pushed $z_2$ **down** (negative Y) and pushed $z_3$ **left** (negative X).
   - They were **repelled** away from each other (the distance between them increased).

---

## Final Result

After just **one single training step**, the total loss dropped from $0.0096$ to a slightly lower number. 

Repeat this process **1,000 times**, and the AI will perfectly arrange these 3 points in 2D so that their pairwise distances perfectly match the original high-dimensional relationships!

---

## Conceptual PyTorch Pseudocode

If you were to implement this in code, the core training loop would look like this:

```python
import torch

# High-D data
X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

# Initial low-D map (random guess)
Z = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], requires_grad=True)

# Hyperparameters
sigma = 1.0
eta = 0.1

# Compute P (high-D similarities) - simplified for this example
# ... (implementation would go here)

# Compute Q (low-D similarities)
# ... (implementation would go here)

# Compute loss
loss = torch.sum(P * torch.log(P / Q))

# Backpropagate
loss.backward()

# Update using SGD
with torch.no_grad():
    Z -= eta * Z.grad

# Zero out gradients for next step
Z.grad.zero_()
