# Learning User Embeddings Through User Post Timelines

We propose a multi-view Social Media user Representation learning framework, which incorporates 

1) temporal activities
2) texts of posts
3) profile information
4) network interactions 

to learn a comprehensive portrait of online users. These features are universal across different social platforms, making our framework very flexible and adaptable. This framework allows us to discover similar users in heterogeneous populations with different beliefs, attitudes, and behaviors, thereby creating new solutions to difficult and important problems.

We first encode the stream of a user's posts as a sequence of triplets composed of(timestamp, textual feature, value), which augments typically limited time series data by incorporating a variety of features within each post. 

We then encode the contextual information of these triplets into an embedding using a transformer-based architecture. We combine this triplet embedding with user profile embedding, and impose two jointly trained objective: 

1) network link prediction to learn interactions between users on social platforms that support such interactions,
2) contrastive learning to pull similar users closer and push dissimilar users farther away. 

Our model learns an embedding space that is aware of user similarity and heterogeneity across temporal, textual, network connection and user profile dimensions. 

The pre-training step described above learns enriched user representations from multiple features, which can be used in unsupervised settings where annotated data is hard to obtain. The method can also be used within supervised learning. By fine-tuning our model on a few supervision data samples, we can adapt it to various downstream tasks.

Example run to train and test:
```
python src/train_and_test.py \
    -m train_test \
    -i data/bert_embeddings_ts_data.pkl \
    -g data/gt_data.pkl \
    -d data/demo_data.pkl \
    -o model_outputs \
```
```
Args:
-m: mode - train_test, train, test
-i: input data path
-g: ground truth binary labels for plotting tsne and classification
-d: profile feature data path
-o: output dir
```