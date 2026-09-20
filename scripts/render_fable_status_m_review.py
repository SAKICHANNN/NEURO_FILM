import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "original": "原图",
    "photometry": "A · 厂家曲线＋正像映射",
    "photometry_optics_proxy": "B · A＋高光扩散近似",
}
SCENES = ["咖啡馆 · 人脸与招牌", "车内 · 暖光皮肤", "花卉 · 花瓣与叶缘", "夜景 · 灯光与反射"]


def render(report_path: Path, destination: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_path = report_path.with_name("visual_review.json")
    rejected = review_path.exists() and json.loads(review_path.read_text(encoding="utf-8")).get("status", "").startswith("STOP")
    rows = report["rows"]
    if [r["source_index"] for r in rows] != list(range(4)):
        raise ValueError("expected four ordered development sources")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def figure(asset, original, label, size=None):
        nonlocal count
        path, source = ROOT / asset["path"], ROOT / original["path"]
        with Image.open(path) as im, Image.open(source) as before:
            if im.size != before.size or (size and list(im.size) != size):
                raise ValueError("comparison geometry differs")
            width, height = im.size
        url = quote(os.path.relpath(path, destination.parent).replace("\\", "/"), safe="/.")
        before_url = quote(os.path.relpath(source, destination.parent).replace("\\", "/"), safe="/.")
        count += 1
        return (f'<figure><figcaption>{html.escape(label)}</figcaption>'
                f'<a href="{url}" target="_blank" rel="noopener"><img loading="eager" '
                f'src="{url}" data-after="{url}" data-before="{before_url}" '
                f'width="{width}" height="{height}" alt="{html.escape(label)}"></a></figure>')

    body = []
    for row in rows:
        index, arms = row["source_index"], row["arms"]
        if set(arms) != set(ARMS):
            raise ValueError("expected original, photometry and optical proxy")
        body.append(f'<section><h2>{index + 1:02d} · {SCENES[index]}</h2><div class="grid">')
        for arm, label in ARMS.items():
            body.append(figure(arms[arm], arms["original"], label, row["size_wh"]))
        body.append('</div><details><summary>查看原尺寸细节对照</summary>')
        for crop in arms["original"]["crops"]:
            body.append(f'<h3>{html.escape(crop["label"])}</h3><div class="grid detail">')
            for arm, label in ARMS.items():
                matches = [x for x in arms[arm]["crops"] if x["label"] == crop["label"]]
                if len(matches) != 1:
                    raise ValueError("missing or duplicate native crop")
                body.append(figure(matches[0], crop, label))
            body.append('</div>')
        body.append('</details></section>')
    if count != 39:
        raise ValueError("expected twelve full images and twenty-seven crops")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>负片曲线 · 固定方案对照</title>
<style>body{margin:0;background:#202020;color:#eee;font:16px/1.6 system-ui,sans-serif}
main{max-width:1700px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:21px}h3{font-size:16px;color:#ccc}
p{max-width:1000px;color:#ccc}section{padding:20px 0;border-top:1px solid #555}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
figure{margin:0}figcaption{padding:10px 0;font-weight:600}img{display:block;width:100%;height:400px;object-fit:contain;background:#171717}
.detail img{height:280px}details{margin-top:20px}summary{cursor:pointer}button{font:inherit;padding:8px 18px;background:#393939;color:#eee;border:1px solid #888;border-radius:6px;cursor:pointer}
.controls{position:sticky;top:0;padding:12px 0;background:#202020f5;z-index:2}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #ebc479;outline-offset:3px}
#errors{color:#ffb8a8}a{color:#ebc479}@media(max-width:760px){main{padding:12px}.grid{grid-template-columns:repeat(3,260px);overflow-x:auto}img{height:300px}}
</style></head><body><main><h1>负片曲线 · 同一原图看变化</h1>
<p><b>胶片类别：</b>彩色电影负片启发的固定试验。曲线来源为 Kodak VISION3 250D；此页不是该型号的准确复现。</p>
<p><b>算法：</b>厂家特性曲线＋通用正像映射；B 额外加入高光扩散近似。没有逐图调参，也没有新增颗粒。先看整图是否有用，再查细节。</p>
<div class="controls"><button id="toggle" aria-pressed="false">全部切换为原图</button><span id="load"></span></div><p id="errors" role="alert"></p>
''' + "".join(body) + '''</main><script>
const images=[...document.images];let showingOriginal=false;
document.querySelector('#toggle').addEventListener('click',()=>{showingOriginal=!showingOriginal;for(const im of images)im.src=im.dataset[showingOriginal?'before':'after'];const b=document.querySelector('#toggle');b.textContent=showingOriginal?'恢复算法结果':'全部切换为原图';b.setAttribute('aria-pressed',String(showingOriginal));});
function status(){const good=images.filter(im=>im.complete&&im.naturalWidth>0).length;document.querySelector('#load').textContent=` 已加载 ${good}/${images.length} 张`;}
for(const im of images){im.addEventListener('load',status);im.addEventListener('error',()=>{document.querySelector('#errors').textContent='有图片加载失败，请勿依据不完整页面作判断。';status();});}status();
</script></body></html>'''
    if rejected:
        page = page.replace('<h1>负片曲线 · 同一原图看变化</h1>', '<h1>失败存档 · 负片曲线固定试验</h1><p><b>本轮未通过：</b>人像肤色偏硬，夜景暗部层次损失。A/B 扩散差异极小；保留完整记录，不作为推荐选项。</p>')
    destination.write_text(page, encoding="utf-8")
    print(json.dumps({"path": str(destination), "images": count, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/fable_status_m_positive_v1/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fable_status_m_review_v1/index.html")
    args = parser.parse_args()
    render(args.report, args.output)
