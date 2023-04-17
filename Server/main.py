import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaForSequenceClassification, RobertaConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import torch.nn.functional as nnf
import itertools
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler
sched = BackgroundScheduler()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

path_config = './model/config.json'
path_model = './model/pytorch_model.bin'
path_bpe = './PhoBERT_base_transformers/bpe.codes'
path_vocab = './PhoBERT_base_transformers/dict.txt'


def get_model(path_model=None, path_config=None, path_bpe=None, path_vocab=None):
    config = RobertaConfig.from_pretrained(
        path_config, from_tf=False, num_labels=6, output_hidden_states=False
    )
    BERT_SA_NEW = RobertaForSequenceClassification.from_pretrained(
        path_model,
        ignore_mismatched_sizes=True,
        config=config
    )

    BERT_SA_NEW.eval()

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes',
                            default=path_bpe,
                            required=False,
                            type=str,
                            help='path to fastBPE BPE'
                            )
        args, unknown = parser.parse_known_args()
        bpe = fastBPE(args)
    except:
        bpe = None

    try:
        vocab = Dictionary()
        vocab.add_from_file(path_vocab)
    except:
        vocab = None
        print('load vocab fail')
    return BERT_SA_NEW, bpe, vocab


model, bpe, vocab = get_model(path_model, path_config, path_bpe, path_vocab)


def predict(model, bpe, sense, vocab):
    subwords = '<s> ' + bpe.encode(sense) + ' </s>'
    encoded_sent = vocab.encode_line(
        subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    encoded_sent = pad_sequences(
        [encoded_sent], maxlen=195, dtype="long", value=0, truncating="post", padding="post")
    mask = [int(token_id > 0) for token_id in encoded_sent[0]]

    encoded_sent = torch.tensor(encoded_sent)
    mask = torch.tensor(mask)
    encoded_sent = torch.reshape(encoded_sent, (1, 195))
    mask = torch.reshape(mask, (1, 195))

    with torch.no_grad():
        outputs = model(encoded_sent,
                        token_type_ids=None,
                        attention_mask=mask)
        logits = outputs[0]
    probs = nnf.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()
    return probs


def retrain():
    print("Executing retrain...")

@app.on_event('startup')
def init():
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain, 'cron', hour=0, minute=0)
    scheduler.start()

@app.get("/predict/{sentence}")
async def predict_sentence(sentence: str):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("INSERT INTO data VALUES (?,?)", (sentence,-1))
    conn.commit()
    conn.close()
    probs = predict(model, bpe, sentence, vocab)
    flat = list(itertools.chain(*probs.tolist()))
    return {
        "Disgust": flat[0],
        "Enjoyment": flat[1],
        "Sadness": flat[2],
        "Fear": flat[3],
        "Anger": flat[4],
        "Other": flat[5]
    }
