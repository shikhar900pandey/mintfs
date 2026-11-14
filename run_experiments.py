import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mintfs import *; from data import *; from config import CFG
from sklearn.ensemble import RandomForestClassifier
torch.set_num_threads(int(os.environ.get("MINTFS_THREADS", "1")))
OUT = os.environ.get("MINTFS_OUT", "results"); os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"
cfg = dict(CFG)
if QUICK: cfg.update(epochs=2, epochs1=3, warm=1, ramp=1, spe=5)

ABL = {"abl_nohyper": dict(use_hyper=False), "abl_nocf": dict(use_cf=False),
       "abl_global": dict(class_specific=False), "abl_noint": dict(use_int=False),
       "abl_nocons": dict(use_cons=False),
       "abl_plain": dict(use_hyper=False, use_cf=False, class_specific=False, use_int=False, use_cons=False)}

def data_for(task):
    if task == "multi": return make_synthetic(COUNTS_MC, binary=False, seed=0), 5
    return make_synthetic([13000, 7800, 3250, 1300 * 1 // 4, 325 // 1][:5] if False else [13000, 780, 325, 130, 65], binary=True, seed=1), 2

def quality(S, info):
    S = set(int(i) for i in S)
    return dict(k=len(S), n_info=len(S & set(info["informative"])), n_weak=len(S & set(info["weak"])),
                n_red=len(S & set(info["redundant"])), n_noise=len(S & set(info["noise"])),
                synergy=bool(set(info["synergy"]) <= S))

def fit_eval(S, tr, C, rare, prior, seed, with_rf=True):
    Xtr, ytr, Xva, yva, Xte, yte = tr
    S = np.array(sorted(int(i) for i in S))
    out = {}
    t = time.time()
    m = train_classifier(Xtr[:, S], ytr, Xva[:, S], yva, C, cfg, seed=seed)
    tau = tune_logit_adjust(m, Xva[:, S], yva, C, prior)
    pt = predict_proba(m, torch.tensor(Xte[:, S]))
    out["hint"] = evaluate(yte, pt, C, rare, prior=prior, tau=tau); out["hint"]["tau"] = tau
    out["hint"]["per_class_f1"] = f1_score(yte, (np.log(np.clip(pt,1e-9,1)) - tau*np.log(prior+1e-8)).argmax(1),
                                          labels=list(range(C)), average=None, zero_division=0).tolist()
    out["hint"]["train_time"] = time.time() - t
    if with_rf:
        rf = RandomForestClassifier(200, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=1,
                                    random_state=seed).fit(Xtr[:, S], ytr)
        out["rf"] = evaluate(yte, rf.predict_proba(Xte[:, S]), C, rare)
    return out

def run(task, seed, kind):
    cfg.update(lam_sparse={"multi": 3.0, "bin": 2.0}[task])
    fn = f"{OUT}/{task}_{seed}_{kind}.json"
    if os.path.exists(fn) and not QUICK: return
    (X, y, info), C = data_for(task)
    tr = split_standardise(X, y, seed)
    Xtr, ytr, Xva, yva, Xte, yte = tr
    counts = np.bincount(ytr, minlength=C); prior = counts / counts.sum(); rare = int(np.argmin(counts))
    res = dict(task=task, seed=seed, kind=kind, counts=counts.tolist(), d=X.shape[1])
    kw = ABL.get(kind, {})
    t = time.time()
    sel, clf, gdet = train_mintfs(Xtr, ytr, Xva, yva, C, cfg, seed=seed, **kw)
    S, g = select_subset(clf, gdet, Xva, yva, C)
    res["select_time"] = time.time() - t
    res["gates"] = g.tolist(); res["S"] = [int(i) for i in S]; res["quality"] = quality(S, info)
    # joint (masked) model without retraining
    m = torch.zeros(X.shape[1]); m[torch.tensor(S)] = 1
    pj = predict_proba(clf, torch.tensor(Xte), m)
    res["joint"] = evaluate(yte, pj, C, rare)
    res["MINT-FS"] = fit_eval(S, tr, C, rare, prior, seed, with_rf=(kind == "main"))
    if kind == "main":
        k = len(S)
        res["No-FS"] = fit_eval(range(X.shape[1]), tr, C, rare, prior, seed)
        t = time.time(); ranks = baseline_rankings(Xtr, ytr, seed); res["rank_time"] = time.time() - t
        res["baselines"] = {}
        for name, rk in ranks.items():
            Sb = rk[:k]
            r = fit_eval(Sb, tr, C, rare, prior, seed); r["quality"] = quality(Sb, info); r["S"] = [int(i) for i in Sb]
            res["baselines"][name] = r
    json.dump(res, open(fn, "w"))
    print("done", task, seed, kind, "k=", len(S), "time %.0f" % (time.time() - t), flush=True)

if __name__ == "__main__":
    wid, nw = int(sys.argv[1]), int(sys.argv[2])
    jobs = []
    for seed in (0, 1, 2):
        jobs += [("multi", seed, "main"), ("bin", seed, "main")]
    for kind in ABL:
        for seed in (0, 1):
            jobs.append(("multi", seed, kind))
        jobs.append(("bin", 0, kind))
    if QUICK: jobs = jobs[:2]
    for i, j in enumerate(jobs):
        if i % nw == wid: run(*j)
