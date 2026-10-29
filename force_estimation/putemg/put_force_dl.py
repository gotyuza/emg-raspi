"""
putEMG-Force の HDF5 データをダウンロードする（公式スクリプトの URL 形式が使えなくなった場合の代わり）。

使い方（PowerShell / コマンドプロンプト）:
  python put_force_dl.py 03 04 05        # 被験者 03, 04, 05 の emg_force データ（全部）
  python put_force_dl.py 03 --list       # ダウンロードせず、対象ファイルの一覧だけ表示
  python put_force_dl.py 05 06 --tasks repeats_long repeats_short --days 1
                                         # 1 日目の repeats_long と repeats_short だけ（容量の節約）
    --tasks : bias / mvc / repeats_long / sequential / repeats_short から選ぶ
    --days  : 1（日付の早い方）/ 2（遅い方）。省略すると両方

保存先: このスクリプトと同じ場所の putemg_force/ フォルダ
"""
import os
import shutil
import sys
import urllib.parse
import urllib.request

TOKEN = "45NY5snj0U4tgQz"
HOST = "https://chmura.put.poznan.pl"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "putemg_force")

# Nextcloud の共有リンクから 1 ファイルを取る URL の形は、サーバーのバージョンで違う。順番に試す。
URL_PATTERNS = [
    HOST + "/public.php/dav/files/" + TOKEN + "/{path}",                                   # 新しい Nextcloud
    HOST + "/s/" + TOKEN + "/download?path={dir}&files={name}",                            # 古い形式（公式スクリプトと同じ）
    HOST + "/index.php/s/" + TOKEN + "/download?path={dir}&files={name}",
]


def candidates(path):
    d, name = os.path.split(path)
    d = "/" + d if d else "/"
    for p in URL_PATTERNS:
        yield p.format(path=urllib.parse.quote(path),
                       dir=urllib.parse.quote(d, safe=""),
                       name=urllib.parse.quote(name))


def fetch(path, dest=None):
    """path（共有フォルダ内のパス）を取得。dest を渡すとファイルに保存、渡さなければ中身を返す"""
    last = None
    for url in candidates(path):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                if dest is None:
                    return r.read()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length", 0))
                    done = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:7.1f} MB", end="")
                os.replace(tmp, dest)
                print()
                return True
        except Exception as e:      # 次の URL の形を試す
            last = f"{url} -> {e}"
    raise RuntimeError("どの URL の形でも取得できませんでした。最後のエラー: " + str(last))


def main():
    argv = sys.argv[1:]
    opt = {"--tasks": [], "--days": []}
    ids, cur = set(), None
    for a in argv:
        if a in opt:
            cur = a
        elif a == "--list":
            cur = None
        elif cur:
            opt[cur].append(a)
        else:
            ids.add(a.zfill(2))
    only_list = "--list" in argv
    if not ids:
        sys.exit("被験者番号を指定してください。例: python put_force_dl.py 01 02")

    print("ファイル一覧(records.txt)を取得中...")
    records = fetch("records.txt").decode("utf-8").split()
    force = [r for r in records if "force" in r]
    targets = [r for r in force if len(r.split("-")) > 1 and r.split("-")[1] in ids]
    if opt["--tasks"]:
        targets = [r for r in targets if r.split("-")[2] in opt["--tasks"]]
    if opt["--days"]:
        keep = []
        for sid in ids:
            mine = [r for r in targets if r.split("-")[1] == sid]
            dates = sorted({"-".join(r.split("-")[3:6]) for r in mine})
            want = {dates[int(d) - 1] for d in opt["--days"] if 0 < int(d) <= len(dates)}
            keep += [r for r in mine if "-".join(r.split("-")[3:6]) in want]
        targets = sorted(keep)
    if not targets:
        print(f"被験者 {sorted(ids)} の emg_force データが見つかりませんでした。")
        print(f"records.txt の行数: {len(records)} / 'force' を含む行: {len(force)}")
        avail = sorted({r.split('-')[1] for r in force if len(r.split('-')) > 1})
        print("力のデータがある被験者番号: " + (" ".join(avail) if avail else "（なし）"))
        print("records.txt の先頭 5 行:")
        for r in records[:5]:
            print("  " + r)
        if force:
            print("'force' を含む行の例:")
            for r in force[:5]:
                print("  " + r)
        sys.exit(1)

    print(f"{len(targets)} ファイルが対象です:")
    for r in targets:
        print("  " + r)
    if only_list:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"ディスクの空き: {shutil.disk_usage(OUT_DIR).free / 1e9:.2f} GB（1 ファイルは数十〜百数十 MB）")
    for r in targets:
        dest = os.path.join(OUT_DIR, r + ".hdf5")
        if os.path.exists(dest):
            print(f"スキップ（取得済み）: {r}")
            continue
        print(f"ダウンロード: {r}")
        fetch("Data-HDF5/" + r + ".hdf5", dest)
    print("完了: " + OUT_DIR)


if __name__ == "__main__":
    main()
