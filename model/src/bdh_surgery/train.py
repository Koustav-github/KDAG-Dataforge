# Trains BDH-GPU models: base pretraining on the pivot corpus, and
# per-direction fine-tuning of clones. Mirrors arXiv:2509.26507 SS7.1's
# protocol: one base per seed, cloned twice, fine-tuned on domain A and B.
import copy

import torch

from .bdh import BDH, TOY
from .domains import make_corpus, make_pivot_corpus

TRAIN_SEQS = 2048
EVAL_SEQS = 256


def train_model(net: BDH, corpus: torch.Tensor, steps: int, lr: float,
                seed: int, batch_size: int = 32) -> list[float]:
    """AdamW with linear decay, following arXiv:2509.26507 Appendix B.4 in spirit."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.1, total_iters=steps)
    net.train()
    history = []
    for _ in range(steps):
        pick = torch.randint(0, corpus.shape[0], (batch_size,), generator=g)
        batch = corpus[pick]
        # BDH.forward's loss path does targets.view(-1), which requires a
        # contiguous tensor; a right-shifted slice like batch[:, 1:] is not
        # contiguous, so materialize both halves before calling the model.
        _, loss = net(batch[:, :-1].contiguous(), targets=batch[:, 1:].contiguous())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        history.append(loss.item())
    net.eval()
    return history


@torch.no_grad()
def evaluate(net: BDH, corpus: torch.Tensor) -> float:
    net.eval()
    _, loss = net(corpus[:, :-1].contiguous(), targets=corpus[:, 1:].contiguous())
    return loss.item()


def train_base(seed: int, steps: int = 400, spec=None) -> BDH:
    """Theta-independent pivot-language base. Cloned by both parents.

    `spec` is a DatasetSpec; None keeps the original baseline behaviour exactly,
    which is what lets the already-trained baseline weights stay valid.
    """
    torch.manual_seed(seed)
    if spec is None:
        net = BDH(TOY)
        corpus = make_pivot_corpus(TRAIN_SEQS, seed=seed)
    else:
        lay = spec.layout
        net = BDH(spec.config())
        corpus = make_pivot_corpus(TRAIN_SEQS, seed=seed, phrase_len=spec.phrase_len,
                                   n_concepts=spec.n_concepts, pivot_base=lay["pivot_base"],
                                   bos=lay["bos_a"], eos=lay["eos"])
    train_model(net, corpus, steps, 1e-3, seed)
    return net


def finetune(base: BDH, lex: list[int], bos: int, seed: int, steps: int = 600,
             spec=None) -> BDH:
    """Clone the base and fine-tune on one translation direction."""
    net = copy.deepcopy(base)
    if spec is None:
        corpus = make_corpus(lex, bos, TRAIN_SEQS, seed=seed)
    else:
        lay = spec.layout
        corpus = make_corpus(lex, bos, TRAIN_SEQS, seed=seed, phrase_len=spec.phrase_len,
                             n_concepts=spec.n_concepts, pivot_base=lay["pivot_base"],
                             sep=lay["sep"], eos=lay["eos"])
    train_model(net, corpus, steps, 5e-4, seed)
    return net
