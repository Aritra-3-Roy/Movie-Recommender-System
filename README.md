# MovieLens Neural Collaborative Filtering (NCF) Recommender

## Overview
This project builds a neural collaborative filtering (NCF) model using TensorFlow and Keras to predict user-movie ratings based on the MovieLens 20M dataset. The model utilizes embeddings for users and movies, followed by a neural network to learn latent interactions between them.

## Dataset
The dataset used in this project is the **MovieLens 20M dataset**, available at [GroupLens](https://grouplens.org/datasets/movielens/).

- The dataset consists of:
  - `ratings.csv` (User ratings of movies)
  - `movies.csv` (Movie metadata)
  - `tags.csv` (User tags for movies)
  - `genome-scores.csv` (Relevance scores between movies and tags)
  - `genome-tags.csv` (Tag descriptions)
  - `links.csv` (Mapping between MovieLens IDs and external movie databases)


## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- scikit-learn


## Model Architecture
- **User and Movie Embeddings**: Each user and movie is represented as a dense vector of size 10, learned during training.
- **Embedding Layers**: Separate embedding layers for users and movies, mapping categorical IDs to dense representations.
- **Flatten Layers**: Converts embeddings into 1D vectors for concatenation.
- **Concatenation Layer**: Merges user and movie embeddings into a single feature vector.
- **Fully Connected Layers**:
  - First dense layer with 1024 neurons and ReLU activation.
  - Output layer with a single neuron for rating prediction.
- **Loss Function**: Mean Squared Error (MSE) to minimize prediction error.
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum for faster convergence.

### Hyperparameters:
| Parameter        | Value  |
|-----------------|--------|
| Embedding Size  | 10     |
| Hidden Layers   | 1024   |
| Batch Size      | 1024   |
| Learning Rate   | 0.08   |
| Momentum        | 0.9    |
| Epochs         | 25     |

## Performance
- The final **Root Mean Squared Error (RMSE)** on the validation set: **0.791**
- Loss trend during training:
  
  ![Loss Plot](loss_plot.png)

## References
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Neural Collaborative Filtering (NCF) Paper](https://arxiv.org/abs/1708.05031)

## Contributor
- **Aritra Roy**

