# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


"""
Example for HypAR, using the Cellphone dataset
"""
import os

import cornac
from cornac.data import Reader, SentimentModality, ReviewModality
from cornac.data.text import BaseTokenizer
from cornac.eval_methods import StratifiedSplit
from cornac.metrics import NDCG, AUC, MAP, MRR, Recall, Precision

from dataset_utils import dataset_converter, load_feedback, load_review, load_sentiment

dataset = 'cellphone'
fpath = os.path.join(os.path.abspath(os.curdir), 'seer-ijcai2020', dataset)


dataset_converter('cellphone')
feedback = load_feedback(os.path.join(fpath, 'ratings.txt'), fmt="UIRT", reader=Reader())
reviews = load_review(os.path.join(fpath, 'review.txt'))
sentiment = load_sentiment(os.path.join(fpath, 'sentiment.txt'), reader=Reader())


# Instantiate an evaluation method to split data into train and test sets.
sentiment_modality = SentimentModality(data=sentiment)

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

eval_method = StratifiedSplit(
    feedback,
    group_by="user",
    chrono=True,
    sentiment=sentiment_modality,
    review_text=review_modality,
    test_size=0.2,
    val_size=0.16,
    exclude_unknowns=True,
    seed=42,
    verbose=True,
    )

# Instantiate the HypAR model, score: 0.205963068063327
hypar = cornac.models.HypAR(
    use_cuda=False,
    stemming=True,
    batch_size=256,
    num_workers=2,
    num_epochs=500,
    early_stopping=25,
    eval_interval=1,
    learning_rate=0.001,
    weight_decay=0.001,
    l2_weight=0.,
    node_dim=64,
    num_heads=3,
    fanout=-1,
    non_linear=True,
    model_selection='best',
    objective='ranking',
    review_aggregator='narre',
    predictor='dot',
    preference_module='lightgcn',
    combiner='concat',
    graph_type='aos',
    num_neg_samples=50,
    layer_dropout=.2,
    attention_dropout=.2,
    user_based=True,
    verbose=True,
    index=0,
    out_path=os.path.abspath(os.curdir),
    learn_explainability=True,
    learn_method='transr',
    learn_weight=0.5,
    embedding_type='ao_embeddings',
    debug=False
)

# Instantiate evaluation measures
metrics = [NDCG(), NDCG(20), NDCG(100), AUC(), MAP(), MRR(), Recall(10), Recall(20), Precision(10), Precision(20)]

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=eval_method, models=[hypar], metrics=metrics,
    user_based=True, verbose=True
).run()
