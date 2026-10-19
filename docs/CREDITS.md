# 图片来源

## `docs/compare.png`

本仓库脚本**合成**的，不是外部素材，无版权风险。

- 源图：`docs/sample.png`
- 生成脚本：`scripts/make_compare_sample.py`
- 放大：用本项目的 `一键放大.bat` 以 x4 / PNG 跑出（512×512 → 2048×2048）
- 拼图脚本：`scripts/make_compare_figure.py`
- 许可：**CC0 1.0 Universal**（公有领域），可自由使用

想复现：

```powershell
python scripts/make_compare_sample.py docs/sample.png
# 把 docs/sample.png 放进 vosr2_input，跑 一键放大.bat -u 4 -f png
python scripts/make_compare_figure.py <源图> <4倍结果> docs/compare.png
```