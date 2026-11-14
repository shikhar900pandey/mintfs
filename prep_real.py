"""Dataset-specific preparation for the three real benchmarks (identifiers / timestamps / label-derived columns removed).
Each loader returns (X, y, feature_names, class_names)."""
import re, numpy as np, pandas as pd
from data import frame_to_xy
__all__ = ['load_wustl_ehms','load_wustl_hdrl','load_ecu_iot','LOADERS','_ecu_features','_finish']

def _finish(df, label, drop):
    X, y, cols = frame_to_xy(df, label, drop)
    names = list(pd.factorize(df.drop_duplicates()[label], sort=True)[1])
    return X, y, cols, names

def load_wustl_ehms(path):
    """WUSTL-EHMS-2020: 3 classes (normal / Spoofing / Data Alteration). Addresses, MACs, ports, packet counter removed."""
    df = pd.read_csv(path, low_memory=False)
    drop = ["Label", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Sport", "Dport", "Dir", "Packet_num"]
    return _finish(df, "Attack Category", drop)

def load_wustl_hdrl(path):
    """WUSTL-HDRL-2024: 5 classes (normal / DDoS / MiTM / Ransomware / Buffer_overflow). Removed: addresses, MACs, ports, device id (IMEI),
    time stamps / boot time and packet counter (each of these separates some attack classes trivially -> leakage)."""
    df = pd.read_csv(path, low_memory=False)
    drop = ["Label", "SrcAddr", "DstAddr", "SrcMac", "DstMac", "Sport", "Dport", "Dir", "IMEI", "RTime", "DTime",
            "Boot_Time_with_date", "Packet_num",
            "svmem_total", "sswap_total", "sdiskusage_total"]   # host-capacity constants identify the device
    return _finish(df, "Attack_categories", drop)

def _ecu_features(df):
    """ECU-IoHT ships only 9 raw columns; per-packet features are derived from Protocol/Length/Info/Time (no addresses, no labels)."""
    import warnings; warnings.filterwarnings("ignore")
    info = df["Info"].astype(str)
    f = pd.DataFrame(index=df.index)
    f["Protocol"] = df["Protocol"].astype(str)
    f["length"] = df["Length"].astype(float)
    t = df["Time"].astype(float)
    f["dt_prev"] = t.diff().fillna(0).clip(lower=0)
    num = lambda pat: pd.to_numeric(info.str.extract(pat)[0], errors="coerce")
    f["src_port"] = num(r"^(?:\[[^\]]*\]\s*)?(\d+)\s+>\s+\d+")
    f["dst_port"] = num(r"^(?:\[[^\]]*\]\s*)?\d+\s+>\s+(\d+)")
    f["tcp_seq"] = num(r"Seq=(\d+)"); f["tcp_ack"] = num(r"Ack=(\d+)"); f["tcp_win"] = num(r"Win=(\d+)")
    f["tcp_len"] = num(r"Len=(\d+)"); f["mss"] = num(r"MSS=(\d+)"); f["ttl"] = num(r"ttl=(\d+)")
    for flag in ["SYN", "ACK", "RST", "PSH", "FIN"]:
        f["flag_" + flag] = info.str.contains(r"\[[^\]]*" + flag + r"[^\]]*\]", regex=True).astype(float)
    for name, pat in [("retransmission", r"Retransmission"), ("dup_ack", r"Dup ACK"), ("no_response", r"no response found"),
                      ("echo_request", r"Echo \(ping\) request"), ("echo_reply", r"Echo \(ping\) reply"),
                      ("arp_who_has", r"Who has"), ("arp_is_at", r" is at "), ("arp_duplicate", r"duplicate use"),
                      ("tls_appdata", r"Application Data"), ("reassembled", r"reassembled PDU"), ("has_tsval", r"TSval")]:
        f[name] = info.str.contains(pat, regex=True).astype(float)
    dst = df["Destination"].astype(str)
    f["dst_broadcast"] = (dst.str.contains("Broadcast") | dst.str.endswith(".255")).astype(float)
    # short-window traffic context (previous 100 packets): sender repetition, protocol mix, length statistics
    codes = pd.factorize(df["Source"].astype(str))[0]
    same = np.zeros(len(df))
    for lag in range(1, 21): same[lag:] += (codes[lag:] == codes[:-lag])
    f["same_src_last20"] = same
    f["len_mean_100"] = f["length"].rolling(100, min_periods=1).mean()
    f["len_std_100"] = f["length"].rolling(100, min_periods=2).std().fillna(0)
    span = (t - t.shift(100)).fillna(t.iloc[min(99, len(t) - 1)] if len(t) else 1.0).clip(lower=1e-6)
    f["rate_100"] = 100.0 / span
    for p in ["ICMP", "TCP", "ARP"]:
        f["frac_" + p + "_100"] = (df["Protocol"].astype(str) == p).astype(float).rolling(100, min_periods=1).mean()
    f["label"] = df["Type of attack"].astype(str)
    return f

def load_ecu_iot(path):
    """ECU-IoHT: 5 classes (No Attack / Smurf / Nmap Port Scan / ARP Spoofing / DoS). `path` may be .xlsx or .csv."""
    df = pd.read_excel(path) if str(path).lower().endswith(("xlsx", "xls")) else pd.read_csv(path, low_memory=False)
    f = _ecu_features(df)
    return _finish(f, "label", [])

LOADERS = {"WUSTL-EHMS-2020": load_wustl_ehms, "WUSTL-HDRL-2024": load_wustl_hdrl, "ECU-IoHT": load_ecu_iot}
