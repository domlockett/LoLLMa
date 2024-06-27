# Introduction to Large Language Models (LLMs)

## What is an LLM?

Large Language Models (LLMs) are sophisticated AI systems designed to understand and generate human-like text based on vast amounts of data. They are trained on diverse datasets containing text from books, articles, websites, and other sources to learn the complexities of language, including grammar, context, and semantic meaning. 
## Tokens

In the context of Large Language Models (LLMs), a token is a fundamental unit of text. Tokens can be as small as single characters or as large as entire words. They are the building blocks that models use to understand and generate language.

Tokens are represented as unique **numerical IDs.** This numerical representation allows the model to process and manipulate the text efficiently.

- **Vocabulary**: A predefined set of tokens, each assigned a unique numerical ID.
- **Tokenization**: The process of converting text into a sequence of tokens. For example, the sentence "Hello, world!" might be tokenized into [15496, 11, 995].

### Tokenization Process

1. **Text Input**: "Hello, world!"
2. **Tokenization**: The text is split into tokens based on the chosen tokenization scheme.
3. **Mapping to IDs**: Each token is converted into its corresponding numerical ID from the vocabulary.
4. **Processing by the Model**: The model processes these numerical IDs to generate a response.
5. **Decoding**: The output IDs are converted back into human-readable text.

Example:

- Input: "Hello, world!"
- Tokens: ["Hello", ",", "world", "!"]
- Token IDs: [15496, 11, 995, 0]


## Parameters

Parameters are the backbone of neural networks, including LLMs. They include weights and biases that the model adjusts during training to learn patterns from data. The number of parameters in a model is indicative of its capacity to represent complex patterns and make accurate predictions.

- **Representation Capacity**: More parameters generally allow the model to capture more intricate patterns in the data.
- **Model Size**: The number of parameters directly affects the model’s size and hardware requirements.
- **Performance**: Larger models with more parameters typically perform better on various tasks, provided they have sufficient training data and computational resources.

## How does an LLM read text?

Imagine a lively party where multiple conversations are happening simultaneously. A scribe (Transformer) and their team of assistants are tasked with understanding and recording all these conversations. Here’s how they do it, step by step:

1. **Embedding Layer** :
 - **Role at the Party** 📝: The scribe and their team begin by learning the language of the party-goers. Every word spoken is translated into a unique representation in their minds.
  - **Function**: This is analogous to the embedding layer in Transformers, which converts input tokens into dense vectors (imagine a csv file where most of the cells have numbers). Each vector attempts to capture the semantic information of the token.

2. **Positional Encoding** :
  - **Role at the Party** ⏳: As the party progresses, the team keeps track of when each conversation happened. The order of events can change the overall narrative.
   - **Function**: Positional encoding adds temporal information to token embeddings, providing the model with the sequence of each token. Without this, the model would not understand the order of words.

3. **Self-Attention Mechanism** :
  - **Role at the Party** 🎯: The team doesn't just record what is said; they also pay attention to who said what and its relation to other conversations. They determine the importance of each person's words based on their relevance to others.
  - **Function**: The self-attention mechanism in Transformers computes a weighted sum of input tokens, where weights are determined by the token's relevance to each other. For each token, self-attention produces three vectors: Query (Q), Key (K), and Value (V). The attention scores are computed as the dot product of the Query and Key vectors, normalized using the softmax function. These scores weight the Value vectors.

   


4. **Multi-Head Attention** :
  - **Role at the Party** 🤹: Each scribe has a team of assistants, each focusing on different aspects of the conversations. One assistant might focus on tone, another on the subject matter, and yet another on the speaker's identity.
  - **Function**: Multi-head attention runs multiple attention mechanisms (heads) in parallel. Each head learns different aspects of token relationships. The outputs of all attention heads are concatenated and linearly transformed to produce the final attention output.

5. **Feed-Forward Network (FFN)** :
- **Role at the Party** 🔄: After listening attentively, the scribe processes what they've heard, applying their own knowledge to interpret the conversations.
 - **Function**: In the Transformer model, this is analogous to the feed-forward network applied to each token. It consists of two linear transformations with a ReLU activation in between.



6. **Residual Connections and Layer Normalization** :
 - **Role at the Party** 🔄⚖️: Throughout the party, the scribes continually revise their notes, adding new insights while preserving the original context.
 - **Function**: Residual connections help to mitigate the vanishing gradient problem, making it easier to train deep networks. Layer normalization stabilizes and speeds up training by normalizing the output of each sub-layer.


### How does the LLM write text?

Once the encoding process is complete, the encoded representation (which now includes both the semantic information and positional information of the input tokens) is passed to the decoder for generating the output. 


1. **Decoder Input Preparation**:
   - **Shifted Right Input**:
     - **Definition**: The target sequence is shifted to the right, adding a `<start>` token at the beginning during training or generating start tokens during inference. This setup helps the model learn to predict the next token.
     - **Purpose**: Enables the model to generate sequences one token at a time, updating its prediction at each step based on the preceding tokens.

2. **Masked Self-Attention**:
   - **Masking Future Tokens**: To prevent the model from seeing future tokens in the sequence (which it should not be aware of during prediction), a mask is applied. This ensures the model generates output in a step-by-step manner, focusing only on the known context.
   - **Self-Attention in Decoder**: Similar to the encoder, the decoder uses a self-attention mechanism to weigh the importance of each token within the target sequence, considering only the past and present tokens (due to the mask).

3. **Encoder-Decoder Attention**:
   - **Cross-Attention**: The decoder attends to the encoder’s output. This cross-attention mechanism allows the decoder to use the context provided by the encoder’s output to generate the next token. The attention scores determine how much focus each token in the input sequence should have on each token in the target sequence.
   - **Weighted Summation**: The attention mechanism computes a weighted sum of the encoder’s outputs, where the weights are derived from the relevance of each token to the current decoder token.

4. **Feed-Forward Network (FFN)**:
   - **Token-Wise Processing**: Similar to the encoder, each token in the decoder sequence is passed through a feed-forward network, which helps in further transforming the representations and applying learned transformations.

5. **Layer Normalization and Residual Connections**:
   - **Maintaining Stability**: The decoder also uses residual connections and layer normalization to maintain stability and improve the learning process by preserving the input features while learning new ones.

6. **Output Generation**:
   - **Softmax Layer**: Finally, the output of the decoder is passed through a softmax layer to produce a probability distribution over the vocabulary.
   - **Token Selection**: The model selects the token with the highest probability as the next word in the sequence. In the case of autoregressive generation, this process repeats, feeding the newly generated token back into the decoder to predict the next token until a stopping condition (like an end-of-sequence token) is met.

This intricate process allows transformers to generate coherent and contextually relevant text by leveraging both the input sequence and previously generated tokens, making them highly effective for tasks like machine translation, text summarization, and conversational AI.
