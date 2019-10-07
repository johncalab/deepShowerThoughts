# deepShowerThoughts

This is a small project I made for the Insight Data Science fellowship. I trained a few different neural networks on reddit's r/showerthoughts, deployed the result as a bot on twitter. twitter.com/deepThoughtsAI

## The bot
The bot is currently hosted on AWS. It updates its status every hour. The status is a combination of AI-generated text, some fixed hashtags, and the three most popular hashtags between New York London San Francisco and Seattle (at the time of posting). The bot also responds to mentions: if you tweet `@deepThoughtsAI`with some text, the bot will respond by trying to complete it to a full sentence. Every minute the bot checks for followers, retweets, likes, and follows users back.

## The AI
The models were trained using PyTorch. The architecture is a simple character-based RNN. The folder ai contains different modules used for training: vocabulary, vectorizer, dataset, a script for generating samples, and one to train the model (using argparse to pass arguments via the a command line). Since I trained models on Google Colab (to take advantage of the free GPU), there is also jupyter notebook aggregating all the modules (this is the most up to date).

### Transformer
I'm currently experimenting with a mini-Transformer architecture. You can find a jupyter notebook for it, but it's still very much a work in progress.

### Workflow
The notebook (or trainer script) works with the following workflow.

- A `DATASET` path is set, and training data is loaded as a pandas dataframe from a csv file contained in the `training_data` folder.
- The boolean flag `RESUME` is set.
- If `RESUME == True`, a vocabulary is loaded from a pickled dictionary, otherwise one is created from the training data.
- If `RESUME == True`, hyperparameters are loaded from a pickled dictionary (`embedding_dim`, `rnn_hidden_dim`, `num_layers`, `dropout_p`, `bidirectional`), otherwise they are set and pickled.
- More hyperparameters are set (`CUDA, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, SPLIT_FRAC`) and training data is loaded in a PyTorch DataLoader object.
- A logger is created (this could be improved).
- While running the training loop, epoch training and validation losses are written to a txt file, so that they may be plotted afterwards.
- Every time the testing loss goes down the model is saved.
- At the end of each epoch a few samples of text are generated to track progress of the model.

## Plots
The plots folder contains a few graphs I used for a presentation I needed to deliver. One plot shows how much I let one model overfit, while the others are concerned with the training data, which comes from reddit's r/showerthoughts.

