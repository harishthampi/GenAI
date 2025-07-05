# Generative AI (Gen AI)

Generative AI (often called Gen AI) is a branch of artificial intelligence focused on creating new content—text, images, videos, music, code, and more—based on patterns learned from vast amounts of data. Rather than just analyzing or classifying information, Gen AI generates original output from prompts or queries. It is a field where already established Large Language Models (LLMs) are used to create AI agents and Retrieval-Augmented Generation (RAG) systems using transformer architecture.

## Key Types of Gen AI Models

- **Transformers / LLMs (Large Language Models)**: Power tools like GPT-4, Gemini, Claude for text generation.
- **Diffusion Models**: Create images by refining noise into structured visuals (e.g., Midjourney, DALL-E).
- **GANs (Generative Adversarial Networks)**: Two neural networks (generator vs. discriminator) competing to produce hyper-realistic content.
- **VAEs (Variational Autoencoders)**: Compress and reconstruct data with randomness, generating variations on the original input.

## Understanding Transformers and LLMs

**Transformers / LLMs** are Large Language Models built using the Transformer architecture. Examples include ChatGPT, Claude, Gemini, DeepSeek, and LLaMA. **ChatGPT** stands for **Chat Generative Pre-trained Transformer**.

To grasp Transformers, it helps to learn the building blocks in this order:

1. **ANN (Artificial Neural Network)**: The most basic neural network; layers of connected neurons for simple prediction tasks.
2. **RNN (Recurrent Neural Network)**: Adds memory; can handle sequences like sentences by remembering previous inputs.
3. **LSTM (Long Short-Term Memory)**: An advanced RNN that solves issues like forgetting older context; better for long sequences.
4. **Transformer Architecture**: Moves beyond RNNs; uses attention mechanisms to capture relationships in the entire sequence all at once, without step-by-step recurrence. This is why Transformers are so powerful for large-scale language modeling.

Transformers are the foundation for today's LLMs. LLMs learn from massive text datasets to generate human-like responses, translate, summarize, and answer questions. To build AI agents and RAG systems, a basic understanding of ANNs is necessary.

## AI, Machine Learning, and Deep Learning

- **AI (Artificial Intelligence)**: Enables computers to perform tasks that typically require human thought, such as problem-solving, learning, reasoning, and understanding language.
- **Machine Learning (ML)**: A subset of AI that develops algorithms allowing computers to learn from data. Instead of following hard-coded instructions, ML models identify patterns and make decisions independently.
- **Deep Learning**: A branch of machine learning focusing on neural networks with many layers (hence “deep”). These layers allow the model to learn highly complex and abstract patterns from large amounts of data. Deep learning excels when you have:
  - Large datasets
  - Complex patterns
  - High computational power

- **Neural Network**: A computational system inspired by the human brain, consisting of layers of nodes (or "neurons") connected in a structure typically divided into an input layer, hidden layers, and an output layer. It’s designed to recognize patterns and learn from data.

- **Universal Approximation Theorem**: Neural networks can approximate any function if they have enough hidden layers and neurons. Thus, deep learning is a **Universal Function Approximator**.

## Categories of Machine Learning

### Supervised Learning
Models are trained on labeled data where the “right answers” are provided. The model learns to map inputs to outputs.

**How it works**:
- You give the model examples with answers.
- It learns to predict the correct output for new, unseen inputs.

**Examples**:
- Email spam detection: Emails labeled spam or not spam.
- Image classification: Photos labeled dog or cat.
- House price prediction: House features + known prices.

**Goal**: Learn to map inputs → outputs.  
**Key Idea**: “Learn from correct answers given during training.”

### Unsupervised Learning
Models find patterns or groupings in data without predefined labels.

**How it works**:
- Finds patterns, groups, or structures in the data without knowing the “correct” answer.

**Examples**:
- Customer segmentation: Grouping similar customers.
- Topic modeling: Finding themes in documents.
- Anomaly detection: Spotting unusual data points.

**Goal**: Discover hidden patterns or structure in data.  
**Key Idea**: “Find patterns without any labels.”

### Reinforcement Learning
Models learn by interacting with an environment and receiving feedback.

**How it works**:
- An agent takes actions in an environment.
- It receives rewards or penalties based on its actions.
- Learns over time to maximize cumulative reward.

**Examples**:
- Playing chess or Go.
- Robot navigation.
- Game-playing AI (Atari, AlphaGo).
- Self-driving cars (simplified simulation).

**Goal**: Learn the best strategy (policy) to get the highest reward.  
**Key Idea**: “Learn by trial and error using feedback from the environment.”

## ANN (Artificial Neural Network)

### General Steps in Building a Neural Network Model

1. **Define Problem Statement**:
   - What are you trying to solve? Classification or regression?
   - Input → Output relationship
   - Business goal or research objective
   - **Example**: Predict if an email is spam or not based on its content.

2. **Collect and Prepare Data**:
   - Gather quality data and make it usable.
   - Collect from sources (CSV, APIs, sensors, etc.).
   - Clean data (remove noise, missing values).
   - Normalize/scale values.
   - Text/Image preprocessing if needed.
   - **Example**: Convert email content into numerical features.

3. **Split Data**:
   - Divide data into:
     - **Training set**: Used to train the model (~70%).
     - **Validation set**: Tune hyperparameters (~10–15%).
     - **Test set**: Final evaluation (~15–20%).

4. **Define Neural Network Architecture**:
   - Choose how your neural network looks.
     - Number of layers (input, hidden, output).
     - Type of layers (Dense, Conv, LSTM, etc.).
     - Activation functions (ReLU, sigmoid, softmax).
     - Loss function & optimizer.

5. **Train Neural Network**:
   - Feed training data to the model and let it learn.
   - Use backpropagation + gradient descent.
   - Track loss and accuracy.
   - Use epochs, batch size, and early stopping.
   - Evaluate performance on the validation set.

6. **Validate Neural Network**: Tune and test on the validation set.
7. **Test Neural Network**: Final unbiased evaluation.
8. **Deploy Neural Network**: Make your model available for real-world use.

### Mathematical Concept Behind Neural Networks

**Linear Regression**: A type of supervised machine learning algorithm that learns from labeled datasets and maps data points with the most optimized linear functions for prediction on new datasets. It assumes a linear relationship between the input and output, meaning the output changes at a constant rate as the input changes. This relationship is represented by a straight line.

#### Example
Let’s say we want to predict salary based on years of experience, modeling the relation as a linear regression. The equation of a line is `y = mx + c`, but for this example, we’ll use `y = wx + b`.

- **y**: Salary (actual output)
- **ȳ**: Predicted salary
- **x**: Years of experience
- **w**: Weight (slope)
- **b**: Bias (intercept)

**Scenario**:
- Salary (`y`) = 20
- Years of Experience (`x`) = 5
- Equation: `ȳ = wx + b`
- Random values: `w = 2`, `b = 5`
- Result: `ȳ = 2(5) + 5 = 15`
- Difference: `y - ȳ = 20 - 15 = 5`

#### Loss/Cost Function
Measures the error between the predicted and actual output. It quantifies how far the model's prediction is from the actual value.

- Loss/Cost: `y - ȳ`
- To simplify and avoid absolute values, we square the error:  
  `Loss/Cost = (y - ȳ)^2`

To minimize the error, we adjust `w` and `b` using differentiation:
- Update rule:  
  `w = w - α * ∂C/∂w`  
  `b = b - α * ∂C/∂b`  
  where `α` is the learning rate.

Using the chain rule to compute derivatives:
- `∂C/∂w = ∂C/∂ȳ * ∂ȳ/∂w`
- `∂C/∂b = ∂C/∂ȳ * ∂ȳ/∂b`

**Loss with respect to prediction**:
- `∂C/∂ȳ = ∂((y - ȳ)^2)/∂ȳ = -2(y - ȳ)` (since `C = (y - ȳ)^2`)
- `∂ȳ/∂w = ∂(wx + b)/∂w = x` (since `ȳ = wx + b`)
- `∂C/∂w = -2x(y - ȳ)`

- `∂ȳ/∂b = ∂(wx + b)/∂b = 1` (since `ȳ = wx + b`)
- `∂C/∂b = -2(y - ȳ)`

**Update rules**:
- `w = w - α(-2x(y - ȳ))` → Weight of the connection in the neural network
- `b = b - α(-2(y - ȳ))` → Bias value to make the function generic

### Training Process of an AI Model (Neural Network)

1. **Initialize Weights and Biases**:
   - Start with random values to break symmetry and let the network learn.
   - **Example**: `w ~ random`, `b ~ random`

2. **Forward Pass (Input → Output)**:
   - Feed input data into the model.
   - Compute predicted output using current weights and biases.

3. **Calculate Loss/Cost**:
   - Measure how wrong the prediction is.
   - Use a loss function (e.g., Mean Squared Error, Cross-Entropy).
   - `Loss = (y - ȳ)^2`

4. **Compute Gradients**:
   - Compute partial derivatives of the loss with respect to weights and biases.
   - Use the Chain Rule (Backpropagation) for efficiency.
   - `∂Loss/∂w`, `∂Loss/∂b`

5. **Update Weights and Biases**:
   - Use gradients to adjust parameters via Gradient Descent.
   - `w = w - α * ∂C/∂w`
   - `b = b - α * ∂C/∂b`
   - `α` = learning rate.

6. **Repeat**:
   - Go back to step 2.
   - Repeat over many epochs (passes over data) until the loss is minimized.

**Goal**: Make predictions closer and closer to true values.