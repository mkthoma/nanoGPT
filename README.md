# Training a Nano GPT from scratch

This repo contains code for training a nano GPT from scratch on any dataset. The implementation is taken from Andrej Karpathy's [repo](https://github.com/karpathy/nanoGPT/tree/master). The live implementation can be found on Huggingface spaces [here](https://huggingface.co/spaces/mkthoma/nanoGPT).

## Hyperparameters
The notebook defines various hyperparameters at the beginning of the script. These hyperparameters control the training process and model architecture. Here's an explanation of some important hyperparameters:

- batch_size: The number of independent sequences processed in parallel during training.
- block_size: The maximum context length for predictions (bigram context).
- max_iters: The maximum number of training iterations.
- learning_rate: The learning rate for the optimizer.
- device: 'cuda' if a GPU is available; otherwise, 'cpu'.
- n_embd: The embedding dimension for tokens.
- n_head: The number of self-attention heads.
- n_layer: The number of Transformer layers.
- dropout: Dropout rate for regularization.

## Model Architecture
The Bigram Language Model is based on the Transformer architecture, which has been widely adopted in natural language processing tasks due to its ability to capture long-range dependencies in sequential data. Here's a detailed explanation of each component in the model:

1. Token and Position Embeddings: 
    - Token Embeddings: This component is responsible for converting input tokens (characters in your case) into continuous vectors. The model uses an embedding table with a size of vocab_size to look up the embeddings for each token. These token embeddings capture the semantic meaning of the characters.

    - Position Embeddings: Transformers don't have built-in information about the order of tokens in the sequence. To address this, position embeddings are added to the token embeddings. These embeddings provide information about the position of tokens in the sequence. Position embeddings are learned during training.

2. Multi-Head Self-Attention Layers:
    - The Multi-Head Self-Attention mechanism allows the model to weigh the importance of different tokens in the sequence when making predictions. It operates in parallel with multiple "heads," each learning different patterns of attention.
    - Key, Query, Value: The input sequence is linearly transformed into three sets of vectors - key, query, and value. These transformations allow the model to learn how to weigh different tokens when making predictions.
    - Scaled Dot-Product Attention: The self-attention mechanism computes an attention score for each token in the sequence with respect to all other tokens. This score is scaled by the square root of the dimension of the key vectors to prevent the gradients from becoming too small or too large.
    - Masking: The model applies a mask to the attention scores to avoid attending to tokens beyond the current position (lower triangular mask). This ensures that the model only uses previous tokens (the bigram context) for making predictions.

3. Feed-Forward Layers: After self-attention, each token's representation goes through a feed-forward neural network. This network consists of a series of linear transformations and non-linear activation functions, such as ReLU. It allows the model to capture complex relationships between tokens.

4. Layer Normalization: Layer normalization is applied after both the self-attention and feed-forward layers. It helps stabilize training by normalizing the activations at each layer. It ensures that the input to each layer has a similar scale, making it easier for the model to learn and generalize.

5. Linear Classification Head: The final layer of the model is a linear transformation that maps the output of the previous layers to a probability distribution over the vocabulary. This is achieved using the lm_head module. The model predicts the probability of the next token given the context.

6. Training and Loss: During training, the model computes the cross-entropy loss between the predicted probability distribution and the actual target token. The loss is used to update the model's parameters through backpropagation and optimization. The model aims to minimize this loss to improve its predictive capabilities.

7. Generating Text: After training, you can use the model to generate text. The generate method takes an initial context (sequence of tokens) and generates new tokens one at a time. It uses the model's learned parameters to make predictions for the next token, effectively generating coherent and contextually relevant text.

This architecture allows the Bigram Language Model to capture complex dependencies in the input data and generate text that follows the patterns and style of the training data. It is a fundamental building block for a wide range of natural language processing tasks and can be further extended and fine-tuned for specific applications.

## Training
To train the model, run the training loop in the script. The training process involves optimizing the model's parameters to minimize the cross-entropy loss. The model is trained on the provided text data, and you can monitor training progress with periodic evaluation of the training and validation loss.

## Training Data
The data used for training the GPT models can be found here:
- [Shakespeare dataset](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt)
- [Wikiedpia dataset](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences)

## Generating Text
After training, you can use the trained model to generate text. The generate method of the model allows you to provide a context (sequence of tokens) and generate additional text based on the learned language patterns. You can control the number of tokens generated by specifying max_new_tokens.