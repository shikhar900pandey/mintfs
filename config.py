CFG = dict(
    alpha=0.5, cap=10.0, r=4,
    h=32, heads=4, L=2, fc=(128, 64, 32), p=(0.2, 0.3, 0.3), spe=30,
    lr=2e-3, wd=1e-4, bs=256, gamma=2.0, sampler="sqrt",
    lam_con=0.3, lam_sup=0.5, patience=8, epochs=30,
    # MINT-FS
    h_sel=32, omega=0.5, lr_gate=2e-2, epochs1=18, warm=3, ramp=5, cf_every=4,
    lam_sparse=2.0, lam_int=2.0, eta=1.0, lam_cons=0.5, B=3,
)
