DSTC8 Meta-Learning User Response Models Task
============================================


Competition Info
--------------

### [2019-07-12] Our competition host Codalab is experiencing an outage.
### [2019-07-15] Our competition host Codalab is back online, registrations are open again. Note if you registered before July 15, 2019 you will have to re-register, we apologize for the inconvenience.

Please sign up for the competition on our [CodaLab](https://aka.ms/dstc8-task2) page, which also contains further details on the timeline, organizers, and terms and conditions.

Task Description
----------------

In goal-oriented dialogue, data is scarce. This is a problem for dialogue
system designers, who cannot rely on large pre-trained models. The aim
of our challenge is to develop natural language generation (NLG) models
which can be quickly adapted to a new domain given a few goal-oriented
dialogues from that domain.

The suggested approach roughly follows the idea of meta-learning 
(e.g. [MAML: Finn, Abbeel, Levine, 2017](https://arxiv.org/abs/1703.03400),
[Antoniou et al. 2018](https://www.bayeswatch.com/2018/11/30/HTYM/),
[Ravi & Larochelle 2017](https://openreview.net/pdf?id=rJY0-Kcll)): During the
training phase, train a model that can be adapted quickly to a new domain:

<p>
<a href="https://github.com/microsoft/dstc8-meta-dialog/raw/docs/img/train-phase.jpg"><img src="https://github.com/microsoft/dstc8-meta-dialog/raw/docs/img/train-phase.jpg" width="500" ></a><br/>
</p>

During the evaluation phase, the model should predict the final user turn of an
incomplete dialogue, given some (houndreds) of examples from the same domain:

<p>
<a href="https://github.com/microsoft/dstc8-meta-dialog/raw/docs/img/test-phase.jpg"><img src="https://github.com/microsoft/dstc8-meta-dialog/raw/docs/img/test-phase.jpg" width="500" ></a><br/>
</p>



Resources
---------

* A large reddit-based dialogue dataset is available. Due to licensing
  restrictions we cannot provide a download, but we provide [code to generate
  the dataset](https://github.com/Microsoft/dstc8-reddit-corpus).
  This database has 1000 subreddits as "domains".

* A smaller goal-oriented dataset split into domains and tasks, named 
  [MetaLWoz](https://www.microsoft.com/en-us/research/project/metalwoz/).
  The dataset contains 37,884 crowd-sourced dialogues divided into 47 domains.
  Each domain is further divided into tasks, for a total task number of 227.
  No annotation is provided aside from the domain labels, task labels, and task
  descriptions.

<p>
<a href="https://github.com/microsoft/dstc8-meta-dialog/raw/docs/img/datasets.jpg"><img src="https://github.com/microsoft/dstc8-meta-dialog/raw/docs/img/datasets.jpg" width="500" ></a><br/>
</p>



Evaluation
----------

Evaluation for this task is using automatic as well as human metrics.

During development, participants can track their progress using word overlap
metrics, e.g. using [nlg-eval](https://github.com/Maluuba/nlg-eval).
Depending on the parameters of `scripts/make_test_set`, you can determine
within-task or across-task generalization within a MetaLWoz domain.

Towards the end of the evaluation phase, we will provide a zip file with
dialogues in a novel domain and a file specifying dialogues and turns that
participants should predict. The file format is the same as the one produced
by `scripts/make_test_set`, each line is a valid JSON object with the following
schema:

```json
{"support_dlgs": ["SUPPORT_DLG_ID_1", "SUPPORT_DLG_ID_2", ...],
 "target_dlg": "TARGET_DLG_ID",
 "predict_turn": "ZERO-BASED-TURN-INDEX"
}
```

Dialogue IDs uniquely identify a dialogue in the provided MetaLWoz zip file.

To generate predictions, condition your (pre-trained) model on the support
dialogues, and use the target dialogue history as context to predict the indicated
user turn.

Make sure that (1) your model has never seen the test domain before predicting
and (2) reset your model before adapting it to the support set and predicting
each dialogue.

On the responses submitted by the participants, we will

1. Run a fixed NLU module to determine whether response intents and
   slots are in line with ground truth.
1. Ask crowd workers to evaluate informativeness and appropriateness of the
   responses.

Submissions should have one response per line, in JSON format, with this schema:

```json
{"dlg_id": "DIALOGUE ID FROM ZIP FILE",
 "predict_turn": "ZERO-BASED PREDICT TURN INDEX",
 "response": "PREDICTED RESPONSE"}
```

where `dlg_id` and `predict_turn` correspond to the `target_dlg` id and
`predict_turn` of the test specification file above, respectively.


Baseline Implementation
-----------------------

A simple retrieval baseline implementation is provided on [github](https://github.com/Microsoft/dstc8-meta-dialog).
The retrieval baseline requires no training; it uses pre-trained embeddings
([BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and a combination of
[SentencePiece](https://github.com/google/sentencepiece/) and 
[FastText](https://github.com/facebookresearch/fastText)). To complete the test
dialogue, the retrieval model returns the response associated with the most
similar dialogue context on the support set, using cosine distance between the
embeddings.

The baseline implementation also implements metabatch iterators that can be
used to train more complex models. Each meta batch is split into domains, each
domain contains a support set and a target set.


Setup
-----

1. install conda / anaconda, e.g. via [miniconda](https://conda.io/miniconda.html)
2. `conda create -n dstc8-baseline python=3.7`
3. `conda activate dstc8-baseline`
4. `conda install -c pytorch pytorch`
5. `python setup.py install -e .`


Running
-------

```bash
# Create sentencepiece and fasttext models, and normalized dialogues
# for both reddit and MetaLWoz datasets
$ ./scripts/preprocess metalwoz metalwoz-v1.zip pp-metalwoz-dir
$ ./scripts/preprocess reddit dstc8-reddit-corpus.zip pp-reddit-dir
```

**Notes:**
1. It's recommended to have 25 GB of space free for the large FastText models and the unzipped dataset dump used to train SentencePiece and FastText.
1. Reddit takes the longest to preprocess, allow 8-24 hours for this script to run end-to-end on it.
1. SentencePiece consumes a ton of memory so the maximum number of lines used to train it is limited to 5 million by default.
   If this is too much for your system, you can set `maxlines` in the script.
1. The normalizers can be found in `mldc/preprocessing/normalization.py`.


Now, create a test set specification file (we will provide an official one later).
This file references dialogues from the zip file by their ID. Each line
specifies a single-domain meta batch, with a support set of size 128 and a
target set of size 1. The (zero-based) index of the turn to predict is also indicated. 
See the evaluation section for more details.

```bash
./scripts/make_test_set ./pp-metalwoz-dir/metalwoz-v1-normed.zip test-spec-cross-task.txt --cross-task
```

Now train the model.  The retrieval model does not actually do any training, so
this step is fast. The infrastructure for training is present in the code, however,
so you can easily add your own methods. Take care to exclude domains for evaluation
(e.g. early stopping) and testing.

Use embedding models trained on reddit to avoid train/test overlap.
Use `--input-embed` to change the embedding type (BERT is default).

```bash
./scripts/baseline retrieval train ./pp-metalwoz-dir/metalwoz-v1-normed.zip \
  --preproc-dir ./pp-reddit-dir --output \
  --output-dir  ./metalwoz-retrieval-model \
  --eval-domain dialogues/ALARM_SET.txt --test-domain dialogues/EVENT_RESERVE.txt
```

Now we run the evaluation on the excluded test domain for the meta batches
specified in `test-spec-cross-task.txt`. This command 

1. Prints the dialogue context, ground truth and predictions
   to standard output, and 
1. Generates files for submitting results and automatic metric evaluation with
   `nlg-eval`.

```bash
./scripts/baseline retrieval predict ./metalwoz-retrieval-model \
    ./pp-metalwoz-dir/metalwoz-v1-normed.zip \
    --test-spec test-spec-cross-task.txt --nlg-eval-out-dir ./out

# calculate metrics (you need to install https://github.com/Maluuba/nlg-eval for that)
nlg-eval --hypothesis ./output-cross-task/EVENT_RESERVE/hyp.txt --references output-cross-task/EVENT_RESERVE/ref.txt
```

Contributing
------------

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
