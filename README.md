# MINT-FS / HINT-Net

Reference implementation of MINT-FS (Minority-Informed Neural Tabular Feature Selection) and HINT-Net
(Hypergraph Interaction Network) for imbalanced binary and multiclass classification.

## Files
- `mintfs.py` - selector, classifier, losses, training and subset extraction
- `data.py` - synthetic benchmarks with known ground truth, stratified splitting and standardisation
- `prep_real.py` - loaders for the three real datasets (identifiers, addresses, MACs, ports, time stamps, device ids and other
  label-leaking columns removed; packet-level features derived for ECU-IoHT)
- `config.py` - hyper-parameters
- `run_experiments.py` - experiment runner (synthetic benchmarks)
- `MINTFS_validation_colab.ipynb` - Colab notebook that runs the real-dataset study and exports the paper tables
- `paper_tables.xlsx` and the `*.json` files - per-dataset results and summary tables behind the paper (5 stratified splits per real dataset)

## Install
    pip install torch numpy scikit-learn pandas scipy openpyxl

## Usage on a real dataset
    from prep_real import LOADERS
    from data import split_standardise
    from mintfs import train_mintfs, select_subset
    from config import CFG
    X, y, cols, classes = LOADERS["ECU-IoHT"]("ECU_IoHT.xlsx")   # or WUSTL-EHMS-2020 / WUSTL-HDRL-2024 (csv)
    tr = split_standardise(X, y, seed=0)
    C = int(y.max()) + 1
    sel, clf, g = train_mintfs(*tr[:4], C, CFG)
    S, g = select_subset(clf, g, tr[2], tr[3], C)

For the full study use the notebook: upload the code files and the dataset files to Colab, set `RUN` in the settings cell
(one dataset per session) and run all cells. The code is CPU-only.

## Datasets (not included; download from the official pages)
- WUSTL-HDRL-2024: https://classes.engineering.wustl.edu/~jain/hdrl/index.html
- WUSTL-EHMS-2020: https://research.engineering.wustl.edu/~jain/ehms/index.html
- ECU-IoHT: https://doi.org/10.25958/5f1f97b837aca
