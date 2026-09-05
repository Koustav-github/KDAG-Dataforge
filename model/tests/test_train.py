import copy
import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.domains import BOS_A, build_lexicons, make_corpus, make_pivot_corpus
from bdh_surgery.train import train_model, train_base, finetune, evaluate


def test_training_reduces_loss():
    torch.manual_seed(0)
    net = BDH(TOY)
    corpus = make_pivot_corpus(256, seed=1)
    history = train_model(net, corpus, steps=60, lr=1e-3, seed=0)
    assert len(history) == 60
    assert sum(history[-5:]) / 5 < sum(history[:5]) / 5


def test_finetune_does_not_mutate_base():
    base = train_base(seed=0, steps=20)
    before = copy.deepcopy(base.state_dict())
    lex_a, _ = build_lexicons(0.5, seed=0)
    finetune(base, lex_a, BOS_A, seed=0, steps=20)
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())


def test_evaluate_returns_positive_scalar():
    net = BDH(TOY)
    assert evaluate(net, make_pivot_corpus(64, seed=2)) > 0
