import torch
import torch.nn as nn
from sparsebit.quantization.quant_model import QuantModel
from sparsebit.quantization.quant_config import _C as default_config
from examples.post_training_quantization.GLUE.CoLA.model import BertEmbeddings


class BertConfig():
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 768
        self.pad_token_id = 0
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.seq_length = 32
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.position_embedding_type = "absolute"


def build_qconfig(changes_list):
    new_config = default_config.clone()
    new_config.defrost()
    new_config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    new_config.merge_from_list(changes_list)
    new_config.freeze()
    return new_config


def test_bert_embeddings():
    model_name = "bert_base_embeddings"
    # the format of list([k1, v1, k2, v2, ...]), ensure config[::2] is key and config[1::2] is value
    qconfig = [
        ("BACKEND", "tensorrt"),
        ("W.QSCHEME", "per-channel-symmetric"),
        ("W.QUANTIZER.TYPE", "uniform"),
        ("W.QUANTIZER.BIT", 8),
        ("W.OBSERVER.TYPE", "MINMAX"),
        ("A.QSCHEME", "per-tensor-symmetric"),
        ("A.QUANTIZER.TYPE", "uniform"),
        ("A.QUANTIZER.BIT", 8),
        ("A.OBSERVER.TYPE", "MINMAX"),
    ]
    qconfig = [j for i in qconfig for j in i]
    qconfig = build_qconfig(qconfig)

    model_config = BertConfig()
    model = BertEmbeddings(model_config)
    model.seq_length = model_config.seq_length

    qmodel = QuantModel(model, qconfig)

    input_ids = torch.zeros(1, 32)
    input_ids[:, :10] = torch.tensor([101,  3533,  2003, 12283,  2084,  1045,  2228,  2984,  1012, 102,])
    input_ids = input_ids.long()
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

    model.eval()
    qmodel.eval()
    out1 = model(input_ids, token_type_ids)
    out2 = qmodel(input_ids, token_type_ids)
    torch.testing.assert_allclose(out1, out2, atol=1e-4, rtol=1e-4)

