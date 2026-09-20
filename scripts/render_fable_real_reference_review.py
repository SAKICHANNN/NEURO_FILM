import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def render_review(manifest_path: Path, destination: Path) -> None:
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = json.loads((ROOT / "outputs/fable_real_reference_development_v1/input_manifest.json").read_text(encoding="utf-8"))
    review_path = manifest_path.parent / "visual_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {"rows": []}
    decisions = {row["source_index"]: row.get("methods", {}) for row in review.get("rows", [])}
    labels = {"simple": "简化颜色拟合", "full": "分组颜色拟合", "tone_only": "仅明暗调整"}
    scenes = ["01 · 咖啡馆：人脸与招牌", "02 · 车内：暖光下的皮肤", "03 · 花卉：花瓣与叶缘", "04 · 夜景：灯光与湿地反射"]
    crop_labels = {"visible_sign": "招牌上可读的字", "left_face": "左侧人脸", "right_face": "右侧人脸与玻璃反光", "face_warm_light": "暖光下的人脸", "hand_and_fabric": "手部与衣料", "petal_detail": "花瓣纹理", "leaf_edges": "叶片边缘", "reflected_color_and_ripples": "水面反射与波纹", "bokeh_highlights": "灯光与光斑"}
    manifest = {"reference": {**inputs["reference"], "path": report["reference"]["path"], "label": "Kodak Portra 400 · 参考照片外观"}, "rows": []}
    for result in report["rows"]:
        index = result["source_index"]
        arms = result.get("arms", {})
        original = arms.get("original", {}).get("path", inputs["sources"][index]["path"])
        methods = [{"id": arm, "label": label, "path": arms.get(arm, {}).get("path"), "status": decisions.get(index, {}).get(arm, "pending") if arm in arms else "rejected"} for arm, label in labels.items()]
        crops = []
        for crop in arms.get("original", {}).get("crops", []):
            paths = {"original": crop["path"]}
            for arm in labels:
                paths[arm] = next((c["path"] for c in arms.get(arm, {}).get("crops", []) if c["label"] == crop["label"]), None)
            crops.append({"label": crop_labels.get(crop["label"], crop["label"]), "paths": paths})
        manifest["rows"].append({"source_index": index, "label": scenes[index], "size_wh": result["size_wh"], "original": original, "methods": methods, "crops": crops})
    rows = manifest["rows"]
    if len(rows) != 4 or len({row["source_index"] for row in rows}) != 4:
        raise ValueError("expected four distinct development sources")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def url(relative):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return quote(os.path.relpath(path, destination.parent).replace("\\", "/"), safe="/.")

    def figure(path, original, label, status, size=None):
        if path is None:
            return f'<figure data-status="rejected"><figcaption>{html.escape(label)}</figcaption><p>本次运行没有有效结果。</p><span class="state"></span></figure>'
        with Image.open(ROOT / path) as image:
            actual_size = image.size
        with Image.open(ROOT / original) as image:
            if image.size != actual_size:
                raise ValueError("comparison geometry differs")
        if size is not None and list(actual_size) != size:
            raise ValueError("saved image dimensions differ from manifest")
        src, before = url(path), url(original)
        return (
            f'<figure data-status="{html.escape(status, quote=True)}">'
            f'<figcaption>{html.escape(label)}</figcaption>'
            f'<a href="{src}" target="_blank" rel="noopener" aria-label="打开{html.escape(label, quote=True)}原尺寸图片">'
            f'<img src="{src}" data-after="{src}" data-before="{before}" '
            f'width="{actual_size[0]}" height="{actual_size[1]}" alt="{html.escape(label, quote=True)}"></a>'
            '<span class="state"></span></figure>'
        )

    ref = manifest["reference"]
    content = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>胶片外观 · 同图对照</title><style>',
        'body{margin:0;background:#202020;color:#ededed;font:16px/1.6 system-ui,sans-serif}',
        'main{max-width:1600px;margin:auto;padding:24px}h1{font-size:28px;margin:0}h2{font-size:21px}',
        'p{color:#c4c4c4;margin:8px 0}a{color:#e9d4a3}button{font:inherit;color:inherit;background:#383838;border:1px solid #777;border-radius:6px;padding:7px 15px;cursor:pointer}',
        'button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #f2c66d;outline-offset:4px}',
        '.reference{display:flex;gap:24px;align-items:center;padding:22px 0}.reference img{width:300px;height:200px;object-fit:contain;background:#171717}',
        '.controls{position:sticky;top:0;background:#202020f5;z-index:2;padding:12px 0;display:flex;gap:20px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #505050}',
        '.grid{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:12px}.scroll{overflow-x:auto}',
        'figure{margin:0;min-width:0}figcaption{font-weight:600;margin:10px 0}figure img{display:block;width:100%;height:330px;object-fit:contain;background:#171717}',
        'figure a{display:block}.state{display:block;font-size:13px;color:#c8b78e;min-height:24px}',
        'section{padding:18px 0 28px;border-bottom:1px solid #4a4a4a}details{margin-top:14px}summary{cursor:pointer;padding:8px 0}',
        '.crops img{height:220px;image-rendering:auto}.crop-title{margin-top:20px}#empty{padding:28px;border:1px solid #686868;margin-top:20px}',
        '[hidden]{display:none!important}.error{color:#ffb6a6}@media(max-width:700px){main{padding:14px}.reference{display:block}.reference img{width:100%;height:220px}.grid{grid-template-columns:repeat(4,240px)}h1{font-size:23px}}',
        '</style></head><body><main><h1>胶片外观 · 同图对照</h1>',
        '<p>先看同一张原图在不同算法下的效果，再检查原尺寸细节。</p>',
        '<div class="reference">',
        f'<a href="{url(ref["path"])}" target="_blank" rel="noopener"><img src="{url(ref["path"])}" alt="胶片外观参考图"></a>',
        f'<div><h2>{html.escape(ref["label"])}</h2><p>参考照片：{html.escape(ref["author"])}</p>',
        f'<p><a href="{html.escape(ref["source_page"], quote=True)}" target="_blank" rel="noopener">原始照片来源</a> · {html.escape(ref["license"])}</p>',
        '<p>按作者报告的胶片类型归类；这里比较参考照片的外观。</p></div></div>',
        '<div class="controls"><button id="toggle" type="button" aria-pressed="false">全部切换为原图</button>',
        '<label><input id="audit" type="checkbox"> 查看完整实验记录（含失败与待审结果）</label>',
        '<span id="count" role="status"></span></div>',
        '<p id="load-status" role="status"></p>',
        '<p id="empty" hidden>当前没有通过视觉筛选的效果。不会把近似原图或严重失真的结果列为推荐。</p>',
    ]
    if report.get("ui_fixture"):
        content.append('<p class="error">界面加载测试夹具：所有图像仅用于检查控件，不是算法效果。</p>')
    for row in rows:
        arms = [{"id": "original", "label": "原图", "path": row["original"], "status": "original"}, *row["methods"]]
        if len(arms) != 4:
            raise ValueError("expected original and three comparison arms")
        content.extend([f'<section data-row="{row["source_index"]}"><h2>{html.escape(row["label"])}</h2>', '<div class="scroll"><div class="grid">'])
        for arm in arms:
            content.append(figure(arm["path"], row["original"], arm["label"], arm["status"], row["size_wh"]))
        content.append('</div></div><details><summary>检查对应位置的细节</summary>')
        for crop in row["crops"]:
            content.extend([f'<div class="crop-title">{html.escape(crop["label"])}</div><div class="scroll"><div class="grid crops">'])
            for arm in arms:
                content.append(figure(crop["paths"][arm["id"]], crop["paths"]["original"], arm["label"], arm["status"]))
            content.append('</div></div>')
        content.append('</details></section>')
    content.append('''<script>
const audit=document.getElementById('audit'),toggle=document.getElementById('toggle');
const query=new URLSearchParams(location.search),voting=query.get('vote')==='1';
audit.checked=voting||query.get('audit')==='1';
const names=voting?{original:'原图',accepted:'待你评价',rejected:'待你评价',pending:'待你评价'}:{original:'原图',accepted:'通过模型视觉筛选，尚非人类偏好结论',rejected:'失败记录 · 不推荐',pending:'待审 · 不推荐'};
function filter(){let count=0;document.querySelectorAll('section').forEach(row=>{
 const accepted=row.querySelector('figure[data-status="accepted"]');row.hidden=!audit.checked&&!accepted;
 if(!row.hidden)count++;row.querySelectorAll('figure').forEach(f=>{f.hidden=!audit.checked&&!['original','accepted'].includes(f.dataset.status);f.querySelector('.state').textContent=names[f.dataset.status]||'未判定';});
});document.getElementById('empty').hidden=count>0;document.getElementById('count').textContent=count+' / 4 张原图';}
audit.addEventListener('change',filter);filter();
if(voting){
 audit.parentElement.hidden=true;
 document.querySelector('h1').textContent='胶片外观 · 由你投票';
 document.querySelector('h1+p').textContent='每行是同一张照片：原图、A 简化版、B 分组版、C 仅明暗。先选你更喜欢的，再看细节；选择会保存在本机浏览器。';
 const key='fable-real-reference-v1-votes';let votes={};try{votes=JSON.parse(localStorage.getItem(key)||'{}');}catch{}
 const choiceLabels=['原图更好','A · 简化版','B · 分组版','C · 仅明暗','都不喜欢／差别太小'];
 document.querySelectorAll('section').forEach(row=>{
  row.querySelectorAll('.grid').forEach(grid=>grid.querySelectorAll('figcaption').forEach((c,i)=>{c.textContent=['原图','A · 简化版','B · 分组版','C · 仅明暗'][i];}));
  const field=document.createElement('fieldset');field.style.cssText='margin-top:16px;border:1px solid #777;padding:12px;display:flex;gap:16px;flex-wrap:wrap';
  const legend=document.createElement('legend');legend.textContent='这一张，你选哪个？';field.append(legend);
  choiceLabels.forEach((name,i)=>{const label=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name='vote-'+row.dataset.row;input.value=String(i);input.checked=votes[row.dataset.row]===i;input.addEventListener('change',()=>{votes[row.dataset.row]=i;localStorage.setItem(key,JSON.stringify(votes));});label.append(input,document.createTextNode(name));field.append(label);});
  row.append(field);
 });
 const exportButton=document.createElement('button');exportButton.textContent='下载我的投票';exportButton.type='button';exportButton.addEventListener('click',()=>{const blob=new Blob([JSON.stringify({comparison:'fable_real_reference_conditional_v1',choices:choiceLabels,votes},null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='film-appearance-votes.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);});document.querySelector('.controls').append(exportButton);
}
toggle.addEventListener('click',()=>{const before=toggle.getAttribute('aria-pressed')!=='true';toggle.setAttribute('aria-pressed',String(before));toggle.textContent=before?'恢复算法结果':'全部切换为原图';document.querySelectorAll('img[data-after]').forEach(img=>{img.src=before?img.dataset.before:img.dataset.after;img.parentElement.href=img.src;});});
const images=[...document.images];let settled=0,failed=0;
function loaded(ok){settled++;if(!ok)failed++;const status=document.getElementById('load-status');status.textContent=failed?'有 '+failed+' 张图片加载失败，请勿据此作比较。':settled===images.length?'全部 '+images.length+' 张参考、结果和细节图片已加载。':'正在加载图片…';status.className=failed?'error':'';}
images.forEach(img=>{if(img.complete)loaded(img.naturalWidth>0);else{img.addEventListener('load',()=>loaded(true),{once:true});img.addEventListener('error',()=>loaded(false),{once:true});}});
</script></main></body></html>''')
    destination.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render_review(args.manifest, args.destination)
