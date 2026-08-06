# FA4 window patch — 規模の照合（論文§5「4 files +296/-13」の出所確定）

- **base = 4f1fde1**（PR#2656 head "Pack-GQA on SM120"）。**upstream 5835c73 は同PRのsquash-merge＝content等価**（5835c73 は本repo objects に無いが 4f1fde1 が pre-squash 同一内容）。∴ vs 4f1fde1 = vs 5835c73。
- 窓commit列: 4611f08(ampere k_chunks)→15c9564(flash_fwd 距離帯dim-prefix窓)→860e2e9(atflash_window.py host表)→c7fd744(interface opt-in)→bc21b79(trace/constexpr compile fix)→[以降 decode split-KV: 0e9a9a2/398961b/46eebce=HEAD]。

| range | files | +/- | 対応 |
|---|---|---|---|
| 4f1fde1..**c7fd744** | 4 | **+296 / -13** | **= 論文§5 の数字（早7月報告のsnapshot・窓ロジック）** ✔完全一致 |
| 4f1fde1..bc21b79 | 4 | +308 / -13 | B1 §6.3 prefill 計測に使ったbinary（bc21b79）。+12 は trace/constexpr の**compilefix・窓ロジック不変** |
| 4f1fde1..HEAD(46eebce) | 4 | +562 / -64 | C2/7B microbench に使ったbinary（prefill窓は同一・decode split-KV が加算） |

## 結論（論文に書くべき数字）
- 論文§5「**4 files, +296/-13**」は **c7fd744 の窓ロジックdiff として正確**（file数4・削除-13 は全snapshotで不変）。
- 厳密に「計測に使ったbinaryの窓diff」を書くなら **+308/-13**（bc21b79・+12 は非意味的 compilefix）。
- **どちらも file数4・-13 は同じ**。修正必須ではない。**推奨**: 「4 files, +296/-13（window logic; measured binary は compile-fix 込みで +308/-13）」の脚注、または +296→+308 に更新のいずれか。
- prefill窓は c7fd744/bc21b79/HEAD で不変（decode split-KV は別機構）。fig6/§6.3 の prefill 速度は bc21b79 由来。
