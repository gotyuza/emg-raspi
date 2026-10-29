"""
【PC 側】モデルを学習して、ラズパイで動かすためのファイルに書き出す。
personal_compare.py と同じフォルダで使う（esn_putemg.py・esn_calcurve.py も必要）。

やること（被験者 1 人ぶん）
  1. 1 日目の repeats_long で学習（personal_compare.py と同じ分け方・同じハイパーパラメータ探し）
  2. 1 日目の repeats_short（テスト）で PC 上の精度と予測値を計算
  3. 次の 3 つを export/ フォルダに書き出す
       model_<モデル>_<被験者>.onnx   … 深層学習モデル本体（ONNX Runtime 用）
       model_<モデル>_<被験者>.npz    … 標準化の値・フィルタ・ESN/ridge の重み・PC での予測値 など
       replay_<被験者>.npz            … テスト用の生の筋電（1kHz）・正解の力（モデルが違っても共通）

対応モデル: transformer（S〜XL）/ lstm（S〜XL）/ esn（S〜L）/ ridge
  cnnlstm（S〜XL）も可（生の筋電の窓を入力にする）

使い方（GPU の PC で）:
  python export_model.py --model transformer --size S --subject 03
  python export_model.py --model lstm --size S --subject 03
  python export_model.py --model esn --size M --subject 03
  python export_model.py --model ridge --subject 03
→ export フォルダの中身をラズパイに送る
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn

import esn_putemg as E
import personal_compare as P
from esn_calcurve import split_blocks


class TransformerONNX(nn.Module):
    """入力 (1, CTX, 特徴量) → 出力 (1, 10)"""
    def __init__(s, m):
        super().__init__(); s.m = m

    def forward(s, x):
        return s.m(x)[0]


class CNNLSTMONNX(nn.Module):
    """1 ステップずつ：直近 150ms の筋電 (1, ch, 150) と状態 h, c → 出力 (1,10) と新しい h, c"""
    def __init__(s, m):
        super().__init__(); s.m = m

    def forward(s, w, h, c):
        e = s.m.enc(w)[:, None, :]
        y, (h2, c2) = s.m.rnn(e, (h, c))
        return s.m.out(y[:, -1]), h2, c2


class LSTMONNX(nn.Module):
    """1 ステップずつ：入力 (1,1,特徴量) と状態 h, c → 出力 (1,10) と新しい h, c"""
    def __init__(s, m):
        super().__init__(); s.m = m

    def forward(s, x, h, c):
        y, (h2, c2) = s.m.rnn(x, (h, c))
        return s.m.out(y[:, -1]), h2, c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="putemg_force")
    ap.add_argument("--band", choices=list(E.BANDS), default="all")
    ap.add_argument("--model", choices=["transformer", "lstm", "cnnlstm", "esn", "ridge"], default="transformer")
    ap.add_argument("--size", default="S")
    ap.add_argument("--subject", default="03")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="export")
    a = ap.parse_args()
    if a.model == "ridge":
        a.size = "S"
    if torch.cuda.is_available():
        P.DEV = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)

    subj = a.subject.zfill(2)
    d1 = P.find_days(a.root)[subj][0]
    t0 = time.time()
    L, S = P.load(d1["repeats_long"], a.band), P.load(d1["repeats_short"], a.band)
    raw_short, _, _, _ = E.load(d1["repeats_short"], a.band)          # フィルタ前の 1kHz 筋電（ラズパイで再生する）
    sign = P.sensor_sign(L)
    sc = dict(mu=L["F"].mean(0), sd=L["F"].std(0) + 1e-8, rsd=L["X"].std(0) + 1e-8)
    fm, vm = split_blocks(len(L["F"]))
    FIT, VAL, TEST = P.view(L, sc, sign, fm), P.view(L, sc, sign, vm), P.view(S, sc, sign)
    Yf = FIT["Y"][FIT["m"]]
    scale = np.maximum(np.percentile(Yf, 99, 0) - np.percentile(Yf, 1, 0), 5.0).astype(np.float32)
    nin, nch = L["F"].shape[1], L["X"].shape[1]
    print(f"被験者 {subj}: 読み込み {time.time() - t0:.0f} 秒 / モデル {a.model} {a.size} / {nch}ch")

    tag = f"{a.model}{'' if a.model == 'ridge' else '_' + a.size}_{subj}"
    bundle = dict(kind=a.model, mu=sc["mu"], sd=sc["sd"], sign=sign, scale=scale, sos=E.SOS,
                  win=E.WIN, hop=E.HOP, fs=E.FS, ctx=P.CTX, washout=E.WASHOUT, nch=nch, band=a.band)
    hp_rng = np.random.default_rng(1234)
    hp_deep = [{k: v[hp_rng.integers(len(v))] for k, v in P.HP_DEEP.items()} for _ in range(a.trials)]
    hp_esn = [{k: v[hp_rng.integers(len(v))] for k, v in P.HP_ESN.items()} for _ in range(a.trials)]

    if a.model == "ridge":
        _, lam, Wo = P.fit_readout(FIT["F"][FIT["m"]], Yf, VAL["F"][VAL["m"]], VAL["Y"][VAL["m"]], scale)
        pred = P.lin(Wo, TEST["F"])
        bundle.update(Wo=Wo)
    elif a.model == "esn":
        n = P.SPEC["esn"][a.size]["n"]
        res = []
        for hp in hp_esn:
            m = P.ESNModel(nin, n, seed=a.seed, **hp); H = m.states(FIT["F"])
            res.append((P.fit_readout(H[FIT["m"]], Yf, H[VAL["m"]], VAL["Y"][VAL["m"]], scale)[0], hp))
        hp = min(res, key=lambda x: x[0])[1]
        m = P.ESNModel(nin, n, seed=a.seed, **hp); H = m.states(FIT["F"])
        _, lam, Wo = P.fit_readout(H[FIT["m"]], Yf, H[VAL["m"]], VAL["Y"][VAL["m"]], scale)
        pred = P.lin(Wo, m.states(TEST["F"]))
        W = m.W.toarray() if hasattr(m.W, "toarray") else m.W
        bundle.update(Win=m.Win, W=W, leak=m.leak, Wo=Wo)
        print(f"  ESN の設定: {hp}  λ={lam:g}")
    else:
        res = []
        for hp in hp_deep:
            torch.manual_seed(a.seed)
            m = P.DEEP[a.model](nin, nch, **P.SPEC[a.model][a.size], drop=hp["drop"]).to(P.DEV)
            v, _ = P.train_deep(m, [FIT], [VAL], hp, a, np.random.default_rng(a.seed), scale)
            res.append((v, hp))
        hp = min(res, key=lambda x: x[0])[1]
        torch.manual_seed(a.seed)
        m = P.DEEP[a.model](nin, nch, **P.SPEC[a.model][a.size], drop=hp["drop"]).to(P.DEV)
        P.train_deep(m, [FIT], [VAL], hp, a, np.random.default_rng(a.seed), scale)
        print(f"  学習の設定: {hp}")
        pred = P.predict(m, TEST) * scale
        m = m.cpu().eval()
        onnx_path = os.path.join(a.out, f"model_{tag}.onnx")
        if a.model == "transformer":
            wrap, args, names, outs = TransformerONNX(m), (torch.zeros(1, P.CTX, nin),), ["x"], ["y"]
        elif a.model == "cnnlstm":
            Lr, Hh = m.rnn.num_layers, m.rnn.hidden_size
            wrap = CNNLSTMONNX(m)
            args = (torch.zeros(1, nch, E.WIN), torch.zeros(Lr, 1, Hh), torch.zeros(Lr, 1, Hh))
            names, outs = ["w", "h", "c"], ["y", "h_out", "c_out"]
            bundle.update(layers=Lr, hidden=Hh, rsd=sc["rsd"])
        else:
            Lr, Hh = m.rnn.num_layers, m.rnn.hidden_size
            wrap = LSTMONNX(m)
            args = (torch.zeros(1, 1, nin), torch.zeros(Lr, 1, Hh), torch.zeros(Lr, 1, Hh))
            names, outs = ["x", "h", "c"], ["y", "h_out", "c_out"]
            bundle.update(layers=Lr, hidden=Hh)
        wrap.eval()
        try:                                       # Transformer の高速処理（ONNX に書き出せない）を止める
            torch.backends.mha.set_fastpath_enabled(False)
        except Exception:
            pass
        if True:
            try:
                torch.onnx.export(wrap, args, onnx_path, input_names=names, output_names=outs,
                                  opset_version=17, dynamo=False)
            except TypeError:
                torch.onnx.export(wrap, args, onnx_path, input_names=names, output_names=outs, opset_version=17)
        print(f"  書き出し: {onnx_path}")

    # PC 上の精度（personal_compare と同じ計算）
    k = TEST["m"]
    ms = P.metrics(TEST["Y"][k], pred[k], scale)
    nr = np.mean([x["nrmse"] for x in ms]); rr = np.nanmean([x["r"] for x in ms])
    print(f"  PC でのテスト精度: NRMSE {nr:.3f} / r {rr:.2f}")

    bundle.update(pred_pc=pred.astype(np.float32), nrmse_pc=nr, r_pc=rr)   # PC での予測値はモデルごとに保存
    np.savez(os.path.join(a.out, f"model_{tag}.npz"), **bundle)
    np.savez(os.path.join(a.out, f"replay_{subj}.npz"), emg=raw_short.astype(np.float32),
             Y=TEST["Y"].astype(np.float32), mask=k)
    print(f"  書き出し: {a.out}/model_{tag}.npz / {a.out}/replay_{subj}.npz")
    print(f"\nラズパイへ送る: scp -r {a.out} pi@<ラズパイのIP>:~/")


if __name__ == "__main__":
    main()
