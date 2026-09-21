import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "L": "局部内容自适应",
    "G": "全图共享响应",
    "GM": "全图响应＋局部支持范围",
    "M": "错位对应关系对照",
}


def render(report_path: Path, destination: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    if len(rows) != 4 or len({str(r["source_id"]) for r in rows}) != 4:
        raise ValueError("Expected four distinct development sources")
    destination.parent.mkdir(parents=True, exist_ok=True)
    body, assets = [], []

    def url(path):
        return quote(os.path.relpath(path, destination.parent).replace("\\", "/"), safe="/.")

    for row in rows:
        sid = html.escape(str(row["source_id"]))
        original = ROOT / row["original_path"]
        with Image.open(original) as image:
            size = image.size
        if list(size) != row["size_wh"] or set(row["arms"]) != set(LABELS):
            raise ValueError(f"Incomplete or mismatched source {sid}")
        choices = {}
        for arm, label in LABELS.items():
            path = ROOT / row["arms"][arm]["image_path"]
            with Image.open(path) as image:
                if image.size != size:
                    raise ValueError(f"Geometry mismatch: {path}")
            choices[arm] = {"url": url(path), "label": label}
            assets.append(str(path.relative_to(ROOT)))
        assets.append(str(original.relative_to(ROOT)))
        data = html.escape(json.dumps(choices, ensure_ascii=False), quote=True)
        body.append(
            f'<section data-choices="{data}"><h2>原图 {sid} · {size[0]} × {size[1]}</h2>'
            '<div class="pair"><figure><figcaption>原图</figcaption>'
            f'<a href="{url(original)}" target="_blank" rel="noopener"><img src="{url(original)}" alt="原图 {sid}"></a></figure>'
            f'<figure><figcaption class="arm-label">{LABELS["L"]}</figcaption>'
            f'<a class="candidate-link" href="{choices["L"]["url"]}" target="_blank" rel="noopener">'
            f'<img class="candidate" src="{choices["L"]["url"]}" alt="原图 {sid} 的局部处理结果"></a></figure></div>'
            '<p class="load-state" role="status"></p>'
            '<details><summary>全部控制组原尺寸文件</summary><ul>'
            + ''.join(f'<li><a href="{value["url"]}" target="_blank" rel="noopener">{value["label"]}</a></li>' for value in choices.values())
            + '</ul></details></section>'
        )
    status = html.escape(str(report.get("status", "未完成质量评审")))
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>局部内容自适应 · 同图对照</title>
<style>body{background:#202020;color:#eee;font:16px/1.6 system-ui;margin:0}main{max-width:1600px;margin:auto;padding:24px}h1{font-size:28px}p{max-width:1100px}section{border-top:1px solid #555;padding:20px 0}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure{margin:0}figcaption{padding:8px 0;font-weight:600}img{display:block;width:100%;height:460px;object-fit:contain;background:#161616}.controls{position:sticky;top:0;z-index:2;background:#202020f5;padding:12px 0}select,button{font:inherit;padding:8px;background:#333;color:white;border:1px solid #888}a{color:#e6c48f}a:focus-visible,select:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #e6c48f;outline-offset:3px}.error{color:#ffad99}details{margin-top:12px}@media(max-width:700px){main{padding:12px}.pair{grid-template-columns:repeat(2,280px);overflow-x:auto}img{height:280px}}</style></head><body><main>
<h1>局部内容自适应 · 同图对照</h1>
<p><b>参考类别：</b>作者标注 Portra 400 的 Italy19 照片外观，未经胶片型号标定。<b>算法：</b>局部内容匹配＋逐图自动拟合颜色响应；没有逐图人工调参，也不是已经训练好的通用预测网络。</p>
<p>左侧始终为原图；右侧统一切换方法。点击图片查看原尺寸。下方结果属于开发实验，是否浓郁舒服、是否保留肤色和文字，需要实际图像评审；数值检查不能替代审美判断。</p>
''' + f'<p>运行记录状态：<strong>{status}</strong>。此页不表示候选已经通过质量评审。</p>' + '''
<div class="controls"><label for="arm">右侧方法：</label><select id="arm">''' + ''.join(f'<option value="{arm}">{label}</option>' for arm, label in LABELS.items()) + '''</select> <button id="original" aria-pressed="false">右侧暂看原图</button></div>
''' + ''.join(body) + '''</main><script>
let showOriginal=false;
const picker=document.querySelector('#arm'),button=document.querySelector('#original');
function update(){for(const section of document.querySelectorAll('section')){const choices=JSON.parse(section.dataset.choices),choice=choices[picker.value],img=section.querySelector('.candidate'),link=section.querySelector('.candidate-link'),original=section.querySelector('img');img.src=showOriginal?original.src:choice.url;link.href=img.src;img.alt=showOriginal?original.alt:choice.label;section.querySelector('.arm-label').textContent=showOriginal?'原图（临时切换）':choice.label}button.setAttribute('aria-pressed',String(showOriginal));button.textContent=showOriginal?'恢复处理结果':'右侧暂看原图'}
picker.onchange=()=>{showOriginal=false;update()};button.onclick=()=>{showOriginal=!showOriginal;update()};
for(const section of document.querySelectorAll('section')){const images=[...section.querySelectorAll('img')],state=section.querySelector('.load-state');const check=()=>{const failed=images.some(i=>i.complete&&i.naturalWidth===0);state.textContent=failed?'图片加载失败，请勿据此判断效果':`已加载 ${images.filter(i=>i.complete&&i.naturalWidth>0).length}/2`;state.className=failed?'load-state error':'load-state'};for(const img of images){img.onload=check;img.onerror=check}check()}
</script></body></html>'''
    destination.write_text(page, encoding="utf-8")
    destination.with_name("viewer_manifest.json").write_text(json.dumps({"report": str(report_path), "assets": assets, "count": len(assets), "quality_pass_claimed": False}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.report, args.destination)
