"""
MINT-FS + HINT-Net : minority-aware hypergraph feature selection and deep
representation learning for imbalanced binary / multiclass tabular data.

Reference implementation accompanying the manuscript.  Pure PyTorch + scikit-learn.
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, balanced_accuracy_score, matthews_corrcoef,
                             recall_score, average_precision_score, mutual_info_score,
                             normalized_mutual_info_score)

DEV = torch.device("cpu")


# ----------------------------------------------------------------------------------
# 1. Class-imbalance characterisation  (Eq. 3-4 of the paper)
# ----------------------------------------------------------------------------------
def class_weights(counts, alpha=0.5, eps=1e-8, cap=10.0):
    """Adaptive class-rarity weights w_c, sum_c w_c = 1, with ratio cap (clipping)."""
    counts = np.asarray(counts, dtype=float)
    pi = counts / counts.sum()
    r = (pi + eps) ** (-alpha)
    r = np.minimum(r, cap * r.min())
    return r / r.sum()


# ----------------------------------------------------------------------------------
# 2. Training-only feature dependency statistics and hypergraph construction
# ----------------------------------------------------------------------------------
def _bin(X, q=5):
    out = np.zeros(X.shape, dtype=np.int64)
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, q + 1)[1:-1]))
        out[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return out


def pair_statistics(Xtr, ytr, C, w, q=5, max_per_class=3000, seed=0):
    """Redundancy R_jk (normalised MI) and class-conditional synergy S_jk
    (Miller-Madow corrected, one-vs-rest, normalised by H(Y_c), weighted by w_c)."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(C):
        ic = np.where(ytr == c)[0]
        idx.append(ic if len(ic) <= max_per_class else rng.choice(ic, max_per_class, replace=False))
    idx = np.concatenate(idx)
    B, y = _bin(Xtr[idx], q), ytr[idx]
    n, d = B.shape
    Red = np.zeros((d, d)); Syn = np.zeros((d, d))
    Hy, Y1 = [], []
    for c in range(C):
        yc = (y == c).astype(int); p = yc.mean()
        Hy.append(-(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12)) + 1e-12); Y1.append(yc)
    single = np.array([[max(mutual_info_score(B[:, j], Y1[c]) - (q - 1) / (2 * n), 0) for j in range(d)]
                       for c in range(C)])
    for j in range(d):
        for k in range(j + 1, d):
            Red[j, k] = Red[k, j] = normalized_mutual_info_score(B[:, j], B[:, k])
            code = B[:, j] * q + B[:, k]
            s = 0.0
            for c in range(C):
                joint = max(mutual_info_score(code, Y1[c]) - (q * q - 1) / (2 * n), 0)
                s += w[c] * max(joint - single[c, j] - single[c, k], 0) / Hy[c]
            Syn[j, k] = Syn[k, j] = s
    Red = Red / (Red.max() + 1e-12)
    Syn = Syn / (Syn.max() + 1e-12)
    return Red, Syn


def build_hypergraph(Red, Syn, r=4):
    """One hyperedge per feature: the feature plus its (r-1) strongest partners under
    D_jk = max(Red_jk, Syn_jk). Duplicated hyperedges are removed. Returns H in {0,1}^{d x M}."""
    d = Red.shape[0]
    r = min(r, d)
    D = np.maximum(Red, Syn); np.fill_diagonal(D, np.inf)
    edges = set()
    for j in range(d):
        edges.add(tuple(sorted(np.argsort(-D[j])[:r].tolist())))
    H = np.zeros((d, len(edges)), dtype=np.float32)
    for m, e in enumerate(sorted(edges)):
        H[list(e), m] = 1.0
    return H


def propagation_matrix(H, w_e):
    """P = Dv^{-1/2} H diag(w_e) De^{-1} H^T Dv^{-1/2}   (HGNN operator).  w_e: (M,) or (C,M)."""
    De = H.sum(0)                                        # (M,)
    if w_e.dim() == 1:
        w_e = w_e[None]
    dv = (H[None] * w_e[:, None, :]).sum(-1).clamp_min(1e-6)  # (C,d)
    A = (H[None] * (w_e / De)[:, None, :]) @ H.t()           # (C,d,d)
    s = dv.pow(-0.5)
    return s[:, :, None] * A * s[:, None, :]


# ----------------------------------------------------------------------------------
# 3. Hard-concrete gates
# ----------------------------------------------------------------------------------
BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1


def hard_concrete(log_alpha, training):
    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / BETA)
    else:
        s = torch.sigmoid(log_alpha)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


def p_open(log_alpha):
    return torch.sigmoid(log_alpha - BETA * math.log(-GAMMA / ZETA))


# ----------------------------------------------------------------------------------
# 4. MINT-FS selector
# ----------------------------------------------------------------------------------
class MINTFS(nn.Module):
    def __init__(self, d, C, H, w, h=32, omega=0.5, use_hyper=True, use_cf=True,
                 class_specific=True, init_logalpha=2.0):
        super().__init__()
        self.d, self.C, self.h = d, C, h
        self.use_hyper, self.use_cf, self.class_specific = use_hyper, use_cf, class_specific
        self.register_buffer("H", torch.tensor(H))
        w = torch.tensor(w, dtype=torch.float32)
        # gate-capacity correction: w_hat_c = clip(w_c / max_c w_c, omega, 1)
        self.register_buffer("w_hat", (w / w.max()).clamp_min(omega))
        self.register_buffer("U", torch.zeros(C, d))
        self.register_buffer("mem", torch.zeros(C, d, h))
        self.phi_w = nn.Parameter(torch.randn(d, h) * 0.3)
        self.phi_b = nn.Parameter(torch.zeros(d, h))
        self.q = nn.Embedding(C, h)
        self.We = nn.Linear(2 * h, h)
        self.edge_mlp = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.Wh = nn.Linear(h, h, bias=False)
        self.score = nn.Sequential(nn.Linear(2 * h + 1, h), nn.GELU(), nn.Linear(h, 1))
        nn.init.constant_(self.score[-1].bias, 0.0)
        nn.init.normal_(self.score[-1].weight, std=0.01)
        # feature-class bias theta_{j,c}: log-odds offset of each gate (as in L0 / STG gates)
        self.theta = nn.Parameter(torch.full((C, d), float(init_logalpha)))

    def class_means(self, x, y):
        E0 = F.gelu(x[:, :, None] * self.phi_w + self.phi_b)            # (B,d,h)
        oh = F.one_hot(y, self.C).float()                                 # (B,C)
        cnt = oh.sum(0)
        Eb = torch.einsum("bc,bdh->cdh", oh, E0) / cnt.clamp_min(1)[:, None, None]
        present = (cnt > 0)[:, None, None]
        if self.training:
            with torch.no_grad():
                self.mem.copy_(torch.where(present, 0.9 * self.mem + 0.1 * Eb.detach(), self.mem))
        return torch.where(present, Eb, self.mem)

    def log_alpha(self, x, y):
        Eb = self.class_means(x, y)                                       # (C,d,h)
        qc = self.q.weight[:, None, :].expand(-1, self.d, -1)
        e = F.gelu(self.We(torch.cat([Eb, qc], -1)))                      # (C,d,h)  e_{j,c}
        if self.use_hyper:
            De = self.H.sum(0)
            pooled = torch.einsum("dm,cdh->cmh", self.H, e) / De[None, :, None]
            a = torch.softmax(self.edge_mlp(pooled).squeeze(-1), dim=1) * self.H.shape[1]   # (C,M)
            P = propagation_matrix(self.H, a)                              # (C,d,d)
            z = F.gelu(torch.einsum("cij,cjh->cih", P, self.Wh(e)))
        else:
            z = torch.zeros_like(e)
        u = self.U[:, :, None] if self.use_cf else torch.zeros(self.C, self.d, 1, device=e.device)
        m = self.score(torch.cat([e, z, u], -1)).squeeze(-1)
        m = m - m.mean(1, keepdim=True)                                   # centred across features: no global drift
        la = self.theta + 3.0 * torch.tanh(m / 3.0)                      # (C,d), bounded
        if not self.class_specific:
            la = la.mean(0, keepdim=True)
        return la

    def gates(self, x=None, y=None, training=True, la=None):
        if la is None:
            la = self.log_alpha(x, y)
        g = hard_concrete(la, training)                                   # (C',d)
        wh = self.w_hat if self.class_specific else torch.ones(1, device=g.device)
        gG = 1 - torch.prod(1 - wh[:, None] * g, dim=0)                   # noisy-OR global gate (d,)
        return gG, la

    def expected_l0(self, la):
        wh = self.w_hat if self.class_specific else torch.ones(1, device=la.device)
        return 1 - torch.prod(1 - wh[:, None] * p_open(la), dim=0)        # (d,)


# ----------------------------------------------------------------------------------
# 5. HINT-Net classifier
# ----------------------------------------------------------------------------------
class HINTBlock(nn.Module):
    def __init__(self, h, heads, p_drop, attn_drop, ffn_mult=2, use_hyper=True):
        super().__init__()
        self.use_hyper = use_hyper
        self.mha = nn.MultiheadAttention(h, heads, dropout=attn_drop, batch_first=True)
        self.ln1 = nn.LayerNorm(h)
        self.Wh = nn.Linear(h, h, bias=False)
        self.Wg = nn.Linear(2 * h, h)
        self.ln2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, ffn_mult * h), nn.GELU(), nn.Dropout(p_drop),
                                 nn.Linear(ffn_mult * h, h))
        self.ln3 = nn.LayerNorm(h)
        self.drop = nn.Dropout(p_drop)

    def forward(self, T, H, w_e):
        a, _ = self.mha(T, T, T, need_weights=False)
        Hl = self.ln1(T + self.drop(a))                                    # local interactions
        if self.use_hyper:
            P = propagation_matrix(H, F.softplus(w_e))[0]                  # (k,k)
            Hh = F.gelu(P @ self.Wh(Hl))                                   # higher-order interactions
            G = torch.sigmoid(self.Wg(torch.cat([Hl, Hh], -1)))
            Hf = self.ln2(G * Hl + (1 - G) * Hh)                           # adaptive multi-scale fusion
        else:
            Hf = Hl
        Hf = self.drop(Hf)
        return self.ln3(Hf + self.drop(self.ffn(Hf)))


class HINTNet(nn.Module):
    def __init__(self, d, C, H, h=64, heads=4, L=2, fc=(256, 128, 64), p=(0.2, 0.3, 0.3),
                 proj=64, attn_drop=0.1, use_hyper=True, use_proto=True):
        super().__init__()
        self.d, self.C, self.use_proto = d, C, use_proto
        self.register_buffer("H", torch.tensor(H))
        self.W = nn.Parameter(torch.randn(d, h) * 0.3)
        self.b = nn.Parameter(torch.zeros(d, h))
        self.eid = nn.Parameter(torch.randn(d, h) * 0.02)
        self.ln0 = nn.LayerNorm(h)
        self.w_e = nn.ParameterList([nn.Parameter(torch.zeros(H.shape[1])) for _ in range(L)])
        self.blocks = nn.ModuleList([HINTBlock(h, heads, p[0], attn_drop, use_hyper=use_hyper) for _ in range(L)])
        self.pool = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.fc1 = nn.Linear(h, fc[0]); self.fc2 = nn.Linear(fc[0], fc[1]); self.fc3 = nn.Linear(fc[1], fc[2])
        self.res = nn.Linear(fc[2], fc[0], bias=False)
        self.ln_r = nn.LayerNorm(fc[0])
        self.d2, self.d3 = nn.Dropout(p[1]), nn.Dropout(p[2])
        self.out = nn.Linear(fc[0], C)
        self.projh = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, proj))
        self.register_buffer("protos", torch.zeros(C, proj))
        self.register_buffer("proto_init", torch.zeros(C, dtype=torch.bool))

    def forward(self, x, gate=None):
        T = self.ln0(x[:, :, None] * self.W + self.b + self.eid)          # adaptive feature tokenisation
        if gate is not None:                                              # MINT-FS gate scales the token
            T = T * gate[None, :, None] if gate.dim() == 1 else T * gate[:, :, None]
        for blk, we in zip(self.blocks, self.w_e):
            T = blk(T, self.H, we)
        a = torch.softmax(self.pool(T).squeeze(-1), dim=1)                # attention pooling
        r = (a[:, :, None] * T).sum(1)
        z = F.normalize(self.projh(r), dim=-1)                            # contrastive embedding
        H1 = F.gelu(self.fc1(r))
        H2 = self.d2(F.gelu(self.fc2(H1)))
        H3 = F.gelu(self.fc3(H2))
        HR = self.d3(self.ln_r(H1 + self.res(H3)))                        # residual FC head
        return self.out(HR), z


# ----------------------------------------------------------------------------------
# 6. Losses
# ----------------------------------------------------------------------------------
def focal_loss(logits, y, wl, gamma=2.0):
    logp = F.log_softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
    return -(wl[y] * (1 - logp.exp()).pow(gamma) * logp).mean()


def proto_losses(model, z, y, wl, Tp=0.1, Ts=0.1, mom=0.99):
    with torch.no_grad():
        for c in y.unique():
            m = F.normalize(z[y == c].mean(0), dim=0)
            if model.proto_init[c]:
                model.protos[c] = F.normalize(mom * model.protos[c] + (1 - mom) * m, dim=0)
            else:
                model.protos[c] = m; model.proto_init[c] = True
    ok = model.proto_init
    s = (z @ model.protos.t()) / Tp
    s = s.masked_fill(~ok[None], -1e4)
    Lp = (wl[y] * F.cross_entropy(s, y, reduction="none")).mean()
    sim = (z @ z.t()) / Ts
    eye = torch.eye(len(y), dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None]) & ~eye
    logprob = sim - torch.logsumexp(sim, 1, keepdim=True)
    npos = pos.sum(1)
    valid = npos > 0
    Ls = -((logprob * pos).sum(1)[valid] / npos[valid]).mean() if valid.any() else z.sum() * 0
    return Lp, Ls


# ----------------------------------------------------------------------------------
# 7. Metrics
# ----------------------------------------------------------------------------------
def gmean(y, p, C):
    r = recall_score(y, p, labels=list(range(C)), average=None, zero_division=0)
    return float(np.prod(np.maximum(r, 1e-6)) ** (1.0 / C))


def evaluate(y, proba, C, rare, tau_pred=None, prior=None, tau=0.0):
    z = np.log(np.clip(proba, 1e-9, 1))
    if tau != 0.0:
        z = z - tau * np.log(prior + 1e-8)
    p = z.argmax(1)
    labs = list(range(C))
    ap = []
    for c in labs:
        if (y == c).any():
            ap.append(average_precision_score((y == c).astype(int), proba[:, c]))
    return dict(macro_f1=f1_score(y, p, labels=labs, average="macro", zero_division=0),
                gmean=gmean(y, p, C), bacc=balanced_accuracy_score(y, p), mcc=matthews_corrcoef(y, p),
                rare_recall=recall_score(y, p, labels=[rare], average="macro", zero_division=0),
                rare_f1=f1_score(y, p, labels=[rare], average="macro", zero_division=0),
                pr_auc=float(np.mean(ap)), acc=float((p == y).mean()))


@torch.no_grad()
def predict_proba(model, X, gate=None, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        lg, _ = model(X[i:i + bs], gate)
        out.append(F.softmax(lg, -1))
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------------
# 8. Training: classifier only (phase 2 / baselines)
# ----------------------------------------------------------------------------------
def sqrt_sampler(ytr, C, n):
    cnt = np.bincount(ytr, minlength=C).astype(float)
    pw = cnt ** (-0.5)                      # class prob proportional to n_c^{0.5}
    w = torch.tensor(pw[ytr], dtype=torch.float32)
    return torch.multinomial(w, n, replacement=True)


def train_classifier(Xtr, ytr, Xva, yva, C, cfg, seed=0, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    counts = np.bincount(ytr, minlength=C)
    w = class_weights(counts, cfg["alpha"], cap=cfg["cap"])
    wl = torch.tensor(w * C, dtype=torch.float32)
    Red, Syn = pair_statistics(Xtr, ytr, C, w, seed=seed)
    H = build_hypergraph(Red, Syn, r=cfg["r"])
    model = HINTNet(d, C, H, h=cfg["h"], heads=cfg["heads"], L=cfg["L"], fc=cfg["fc"], p=cfg["p"],
                    use_hyper=cfg.get("use_hyper", True))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    E = cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xva)
    best, best_state, bad = -1, None, 0
    N = len(yt)
    n_ep = min(N, cfg["spe"] * cfg["bs"])
    for ep in range(E):
        model.train()
        idx = sqrt_sampler(ytr, C, n_ep) if cfg["sampler"] == "sqrt" else torch.randperm(N)[:n_ep]
        for i in range(0, n_ep, cfg["bs"]):
            b = idx[i:i + cfg["bs"]]
            if len(b) < 8: continue
            lg, z = model(Xt[b])
            L = focal_loss(lg, yt[b], wl, cfg["gamma"])
            if cfg["lam_con"] > 0:
                Lp, Ls = proto_losses(model, z, yt[b], wl)
                L = L + cfg["lam_con"] * (Lp + cfg["lam_sup"] * Ls)
            opt.zero_grad(); L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        pv = predict_proba(model, Xv)
        f1 = f1_score(yva, pv.argmax(1), labels=list(range(C)), average="macro", zero_division=0)
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    model.load_state_dict(best_state)
    return model


def tune_logit_adjust(model, Xva, yva, C, prior, grid=(0.0, 0.25, 0.5, 1.0)):
    pv = predict_proba(model, torch.tensor(Xva))
    best, bt = -1, 0.0
    for t in grid:
        z = np.log(np.clip(pv, 1e-9, 1)) - t * np.log(prior + 1e-8)
        f = f1_score(yva, z.argmax(1), labels=list(range(C)), average="macro", zero_division=0)
        if f > best + 1e-4: best, bt = f, t
    return bt


# ----------------------------------------------------------------------------------
# 9. Phase 1: joint MINT-FS + HINT-Net training
# ----------------------------------------------------------------------------------
@torch.no_grad()
def counterfactual_utility(selector, model, X, y, ref, C, per_class=300, seed=0):
    """U_{j,c} = E_{x:y=c}[ loss(x with feature j replaced by ref_j) - loss(x) ]   (all gates forced open)."""
    rng = np.random.RandomState(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], min(per_class, (y == c).sum()), replace=False)
                          for c in range(C)])
    Xs, ys = torch.tensor(X[idx]), torch.tensor(y[idx])
    model.eval()
    ones = torch.ones(X.shape[1])
    base = F.cross_entropy(model(Xs, ones)[0], ys, reduction="none")
    U = torch.zeros(C, X.shape[1])
    for j in range(X.shape[1]):
        Xm = Xs.clone(); Xm[:, j] = ref[j]
        dl = F.cross_entropy(model(Xm, ones)[0], ys, reduction="none") - base
        for c in range(C):
            U[c, j] = dl[ys == c].mean()
    U = U / (U.abs().max(1, keepdim=True).values + 1e-8)
    return U


def train_mintfs(Xtr, ytr, Xva, yva, C, cfg, seed=0, use_hyper=True, use_cf=True, class_specific=True,
                 use_int=True, use_cons=True, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    d = Xtr.shape[1]
    counts = np.bincount(ytr, minlength=C)
    w = class_weights(counts, cfg["alpha"], cap=cfg["cap"])
    wl = torch.tensor(w * C, dtype=torch.float32)
    Red, Syn = pair_statistics(Xtr, ytr, C, w, seed=seed)
    H = build_hypergraph(Red, Syn, r=cfg["r"])
    sel = MINTFS(d, C, H, w, h=cfg["h_sel"], omega=cfg["omega"], use_hyper=use_hyper, use_cf=use_cf,
                 class_specific=class_specific)
    clf = HINTNet(d, C, H, h=cfg["h"], heads=cfg["heads"], L=cfg["L"], fc=cfg["fc"], p=cfg["p"],
                  use_hyper=use_hyper)
    theta = [sel.theta]
    others = [p for n, p in sel.named_parameters() if n != "theta"]
    opt = torch.optim.AdamW([{"params": clf.parameters(), "lr": cfg["lr"], "weight_decay": cfg["wd"]},
                             {"params": others, "lr": cfg["lr"], "weight_decay": cfg["wd"]},
                             {"params": theta, "lr": cfg["lr_gate"], "weight_decay": 0.0}])
    E = cfg["epochs1"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    Rm, Sm = torch.tensor(Red, dtype=torch.float32), torch.tensor(Syn, dtype=torch.float32)
    triu = torch.triu(torch.ones(d, d), 1)
    Xt, yt, Xv = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xva)
    ref = torch.tensor(np.median(Xtr, 0), dtype=torch.float32)
    N = len(yt)
    n_ep = min(N, cfg["spe"] * cfg["bs"])
    snapshots, best, best_pack = [], -1, None
    for ep in range(E):
        lam_sp = cfg["lam_sparse"] * min(1.0, max(0.0, (ep - cfg["warm"]) / max(1, cfg["ramp"])))
        if use_cf and ep >= cfg["warm"] - 1 and (ep - (cfg["warm"] - 1)) % cfg["cf_every"] == 0:
            sel.U.copy_(counterfactual_utility(sel, clf, Xtr, ytr, ref, C, seed=seed + ep))
        clf.train(); sel.train()
        idx = sqrt_sampler(ytr, C, n_ep) if cfg["sampler"] == "sqrt" else torch.randperm(N)[:n_ep]
        for i in range(0, n_ep, cfg["bs"]):
            b = idx[i:i + cfg["bs"]]
            if len(b) < 8: continue
            xb, yb = Xt[b], yt[b]
            la = sel.log_alpha(xb, yb)
            gG, _ = sel.gates(training=True, la=la)
            lg, z = clf(xb, gG)
            L = focal_loss(lg, yb, wl, cfg["gamma"])
            Lp, Ls = proto_losses(clf, z, yb, wl)
            L = L + cfg["lam_con"] * (Lp + cfg["lam_sup"] * Ls)
            if lam_sp > 0:
                L = L + lam_sp * sel.expected_l0(la).mean()
            if use_int:
                Gm = gG[:, None] * gG[None]
                Lint = ((Rm * Gm * triu).sum() - cfg["eta"] * (Sm * Gm * triu).sum()) / (d * (d - 1) / 2)
                L = L + cfg["lam_int"] * Lint
            if use_cons:
                gs = torch.stack([sel.gates(training=True, la=la)[0] for _ in range(cfg["B"])])
                L = L + cfg["lam_cons"] * ((gs - gs.mean(0)) ** 2).sum(1).mean() / d
            opt.zero_grad(); L.backward()
            nn.utils.clip_grad_norm_(list(clf.parameters()) + list(sel.parameters()), 1.0); opt.step()
            with torch.no_grad():
                sel.theta.clamp_(-8.0, cfg.get("theta_max", 3.0))   # keep gates out of the saturated, zero-gradient regime
        sched.step()
        with torch.no_grad():
            sel.eval()
            # deterministic gate for validation: use validation-free class means from training memory
            gdet, _ = sel.gates(training=False, la=sel.log_alpha(Xt[:2048], yt[:2048]))
        pv = predict_proba(clf, Xv, gdet)
        f1 = f1_score(yva, pv.argmax(1), labels=list(range(C)), average="macro", zero_division=0)
        if ep >= int(0.6 * E) and f1 > best:
            best, best_pack = f1, (gdet.clone(), {k: v.clone() for k, v in clf.state_dict().items()})
        if verbose:
            print(f"ep{ep:02d} valF1={f1:.3f} k(g>=.5)={(gdet>=.5).sum().item()} lam_sp={lam_sp:.3f}")
    gdet, state = best_pack
    clf.load_state_dict(state)
    return sel, clf, gdet


def select_subset(clf, gdet, Xva, yva, C, taus=(0.2, 0.4, 0.6, 0.8), tol=0.01, kmin=3):
    """Validation-guided threshold: the smallest subset whose hard-masked validation macro-F1
    is within `tol` of the best one."""
    Xv = torch.tensor(Xva)
    res = []
    for t in taus:
        m = (gdet >= t).float()
        if m.sum() < kmin:
            m = torch.zeros_like(gdet); m[torch.topk(gdet, kmin).indices] = 1
        pv = predict_proba(clf, Xv, m)
        res.append((int(m.sum()), f1_score(yva, pv.argmax(1), labels=list(range(C)), average="macro",
                                           zero_division=0), m))
    bf = max(r[1] for r in res)
    ok = [r for r in res if r[1] >= bf - tol]
    k, f, m = min(ok, key=lambda r: r[0])
    return np.where(m.numpy() > 0)[0], gdet.numpy()


# ----------------------------------------------------------------------------------
# 10. Baseline feature selectors (top-k)
# ----------------------------------------------------------------------------------
def baseline_rankings(Xtr, ytr, seed=0):
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    ranks = {}
    f, _ = f_classif(Xtr, ytr); ranks["ANOVA-F"] = np.argsort(-np.nan_to_num(f))
    mi = mutual_info_classif(Xtr, ytr, random_state=seed); ranks["MI"] = np.argsort(-mi)
    rf = RandomForestClassifier(200, class_weight="balanced_subsample", n_jobs=2, random_state=seed,
                                min_samples_leaf=2).fit(Xtr, ytr)
    ranks["RF-imp"] = np.argsort(-rf.feature_importances_)
    lr = LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=300, class_weight="balanced").fit(Xtr, ytr)
    ranks["L1-LR"] = np.argsort(-np.abs(lr.coef_).sum(0))
    # mRMR (MID scheme) using MI relevance and |Spearman|-free NMI redundancy on binned features
    B = _bin(Xtr); d = Xtr.shape[1]
    sub = np.random.RandomState(seed).choice(len(ytr), min(6000, len(ytr)), replace=False)
    red = np.array([[normalized_mutual_info_score(B[sub, j], B[sub, k]) if j < k else 0 for k in range(d)]
                    for j in range(d)]); red = red + red.T
    sel = [int(np.argmax(mi))]
    while len(sel) < d:
        rest = [j for j in range(d) if j not in sel]
        sc = [mi[j] - red[j, sel].mean() for j in rest]
        sel.append(rest[int(np.argmax(sc))])
    ranks["mRMR"] = np.array(sel)
    return ranks


def kuncheva(a, b, d):
    k = len(a)
    if k == 0 or k == d: return 1.0
    r = len(set(a) & set(b))
    return (r * d - k * k) / (k * (d - k))
