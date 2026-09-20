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
    "candidate": "Ektar 100 → Portra Endura · 固定光谱链",
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
        index = row["source_index"]
        arms = {name: row[name] for name in ARMS}
        if set(arms) != set(ARMS):
            raise ValueError("expected original and fixed spectral candidate")
        body.append(f'<section><h2>{index + 1:02d} · {SCENES[index]}</h2><div class="grid">')
        for arm, label in ARMS.items():
            body.append(figure(arms[arm], arms["original"], label, row["size_wh"]))
        body.append('</div>')
        if index == 3:
            body.append('<p role="note"><strong>明确缺陷：</strong>背景栏杆接近消失，暗反射层次明显损失。本图内容保持未通过；保留展示供判断外观方向。</p>')
        body.append('<details><summary>查看原尺寸细节对照</summary>')
        for crop in arms["original"]["crops"]:
            body.append(f'<h3>{html.escape(crop["label"])}</h3><div class="grid detail">')
            for arm, label in ARMS.items():
                matches = [x for x in arms[arm]["crops"] if x["label"] == crop["label"]]
                if len(matches) != 1:
                    raise ValueError("missing or duplicate native crop")
                body.append(figure(matches[0], crop, label))
            body.append('</div>')
        body.append('</details>')
        body.append(f'<fieldset data-scene="{index}"><legend>这张结果的外观</legend>' + ''.join(f'<label><input type="radio" name="vote-{index}" value="{value}"> {label}</label> ' for value, label in [('keep','有胶片味，保留'),('weak','差异或胶片味不够'),('uncomfortable','过浓或不舒服'),('content','细节损失，否决')]) + '</fieldset></section>')
    if count != 26:
        raise ValueError("expected eight full images and eighteen crops")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ektar 光谱链 · 原图对照</title>
<style>body{margin:0;background:#202020;color:#eee;font:16px/1.6 system-ui,sans-serif}
main{max-width:1700px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:21px}h3{font-size:16px;color:#ccc}
p{max-width:1000px;color:#ccc}fieldset{margin-top:18px;border:1px solid #777}fieldset label{display:inline-block;margin:6px 18px 6px 0}section{padding:20px 0;border-top:1px solid #555}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
figure{margin:0}figcaption{padding:10px 0;font-weight:600}img{display:block;width:100%;height:400px;object-fit:contain;background:#171717}
.detail img{height:280px}details{margin-top:20px}summary{cursor:pointer}button{font:inherit;padding:8px 18px;background:#393939;color:#eee;border:1px solid #888;border-radius:6px;cursor:pointer}
.controls{position:sticky;top:0;padding:12px 0;background:#202020f5;z-index:2}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #ebc479;outline-offset:3px}
#errors{color:#ffb8a8}a{color:#ebc479}@media(max-width:760px){main{padding:12px}.grid{grid-template-columns:repeat(2,280px);overflow-x:auto}img{height:300px}}
</style></head><body><main><h1>彩色负片 · Ektar 光谱链对照</h1>
<p><b>胶片类别：</b>Ektar 100 负片模型 → Portra Endura 相纸模型。外部光谱先验对照，未经真实型号校准。</p>
<p><b>算法：</b>SpektraFilm 完整负片—相纸—扫描光度链。固定 EV 0，无自动曝光、逐图调色或 LUT 捷径；本轮未启用光晕与颗粒。来源是显示 sRGB 照片，不能等同原始场景光。</p>
<p><a href="https://github.com/andreavolpato/spektrafilm">算法来源：SpektraFilm</a> · 仅内部研究对照，不作为训练标签。</p>
<p>内部观察：这四张都有可见变化；暖光肤色进一步偏红，夜景背景结构接近消失，暗部层次明显损失。本轮尚未通过内容保持；请评价外观方向。</p>
<div class="controls"><button id="toggle" aria-pressed="false">全部切换为原图</button><span id="load"></span></div><p id="errors" role="alert"></p>
''' + "".join(body) + '''</main><script>
const voteKey='fable-ektar-external-run3-votes';
let savedVotes={};try{savedVotes=JSON.parse(localStorage.getItem(voteKey)||'{}')}catch{}
for(const field of document.querySelectorAll('fieldset[data-scene]')){for(const input of field.querySelectorAll('input')){input.checked=savedVotes[field.dataset.scene]===input.value;input.addEventListener('change',()=>{savedVotes[field.dataset.scene]=input.value;try{localStorage.setItem(voteKey,JSON.stringify(savedVotes))}catch{document.querySelector('#errors').textContent='浏览器不能保存投票，请直接在对话中告诉我。'}})}}
const images=[...document.images];let showingOriginal=false;
document.querySelector('#toggle').addEventListener('click',()=>{showingOriginal=!showingOriginal;for(const im of images)im.src=im.dataset[showingOriginal?'before':'after'];const b=document.querySelector('#toggle');b.textContent=showingOriginal?'恢复算法结果':'全部切换为原图';b.setAttribute('aria-pressed',String(showingOriginal));});
function status(){const good=images.filter(im=>im.complete&&im.naturalWidth>0).length;document.querySelector('#load').textContent=` 已加载 ${good}/${images.length} 张`;}
for(const im of images){im.addEventListener('load',status);im.addEventListener('error',()=>{document.querySelector('#errors').textContent='有图片加载失败，请勿依据不完整页面作判断。';status();});}status();
</script></body></html>'''
    if rejected:
        page = page.replace('<h1>彩色负片 · Ektar 光谱链对照</h1>', '<h1>失败存档 · 固定光谱链</h1><p><b>本轮未通过：</b>本轮未通过内部外观或内容检查；保留完整记录，不作为推荐选项。</p>')
    destination.write_text(page, encoding="utf-8")
    print(json.dumps({"path": str(destination), "images": count, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/fable_ektar_external_control_v1_run3/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fable_ektar_external_review_v1/index.html")
    args = parser.parse_args()
    render(args.report, args.output)
