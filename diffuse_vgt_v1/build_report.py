"""CPU-only evidence-to-report renderer. Never evaluates a model or reads GT.

Usage: python3 build_report.py [--check-only]
Existing reports are refused, never overwritten. Final independent review is
quoted as evidence, not invented by this report generator.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TEMPLATE = Path.home() / '.codex/templates/report-html-canonical/canonical-exemplar.html'
MODELS = ('baseline', 'O', 'A', 'B', 'C')
MODES = ('real_single', 'global', 'sigmoid', 'blob', 'lowfreq')
CELLS = tuple(f'{s}/{g}/{m}' for s in ('val', 'test') for g in ('all', 'single', 'multi')
              for m in ('patch_pooled', 'patch_image_balanced', 'pixel_image_balanced'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def public_safe(text):
    """Reject sensitive source text intact; never silently edit review verdicts."""
    patterns = (r'/(?:Users|Volumes|home|root|private|mnt|tmp)/', r'~[/\\]',
                r'[A-Za-z]:[\\/]', r'[\w.+-]+@[\w.-]+',
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b', r'\b(?:localhost|[\w.-]+\.(?:local|internal|example))\b',
                r'(?i)\b(?:ssh|sftp|file)://', r'(?i)\b(?:password|api[_-]?key|access[_-]?token|secret)\s*[:=]')
    require(not any(re.search(pattern, text) for pattern in patterns),
            'Private content rejected; prepare an explicitly public-safe source review/field without changing conclusions')
    return text


def load(root):
    # Gate before reading any metric payload: no partial report on failed runs.
    for path in [root / 'results/failure.json', *(root / f'runs/{a}_seed0/failure.json' for a in 'ABC')]:
        require(not path.exists(), f'Failure evidence exists: {path}')
    done = {a: read(root / f'runs/{a}_seed0/completion.json') for a in 'ABC'}
    frozen = read(root / 'frozen/manifest.json')
    freeze_hash = sha(root / 'frozen/manifest.json')
    for arm, record in done.items():
        require(record['status'] == 'training_complete' and record['arm'] == arm
                and record['steps'] == frozen['steps'] == 69600
                and record['freeze_manifest_sha256'] == freeze_hash, f'Incomplete/stale arm {arm}')
    selection = read(root / 'development_selection.json')
    require(selection['status'] == 'development_selected_real_evaluation_not_run'
            and selection['freeze_manifest_sha256'] == freeze_hash, 'Incomplete/stale development selection')
    require(selection['tie_order'] == list(MODELS[1:]), 'Changed tie order')
    for arm in MODELS[1:]:
        means = selection['results'][arm]['means']
        require(set(means) == set(MODES) and all(finite(v) and v > 0 for v in means.values()), 'Invalid development means')
        require(finite(selection['scores'][arm]), 'Invalid development score')
    winner = min(MODELS[1:], key=lambda a: selection['scores'][a])
    require(selection['winner'] == winner, 'Winner conflicts with frozen score/order')
    summary = read(root / 'results/summary.json')
    require(summary['status'] == 'complete' and summary['metric_cells'] == 18, 'Evaluation incomplete')
    require(summary['development_selection_sha256'] == sha(root / 'development_selection.json')
            and summary['freeze_manifest_sha256'] == freeze_hash
            and summary['frozen_winner'] == winner, 'Stale evaluation binding')
    require(summary['evaluation_manifest_sha256'] == sha(root / 'results/evaluation_manifest.json'), 'Changed evaluation manifest')
    require(read(root / 'results/evaluation_manifest.json')['status'] == 'complete', 'Manifest incomplete')
    for arm in 'ABC':
        require(summary['completion_evidence_sha256'][arm]['completion.json'] == sha(root / f'runs/{arm}_seed0/completion.json'), 'Changed completion')
        history_sha = sha(root / f'runs/{arm}_seed0/history.jsonl')
        require(history_sha == summary['completion_evidence_sha256'][arm]['history.jsonl']
                == done[arm]['outputs']['history.jsonl'], 'Changed history binding')
    for name in ('per_image_deltas.csv', 'per_scene_deltas.csv'):
        path = root / 'results' / name
        require(path.is_file() and sha(path) == summary['paired_artifacts_sha256'][name], 'Missing/changed paired evidence')
    require(set(summary['metrics']) == set(MODELS), 'Incomplete model cohort')
    for values in summary['metrics'].values():
        require(set(values) == set(CELLS) and all(finite(v) and v >= 0 for v in values.values()), 'Invalid 18-cell table')
    passed = winner in 'ABC' and all(summary['metrics'][winner][k] < summary['metrics']['baseline'][k] for k in CELLS)
    require(summary['winner_passes_seed0_gate'] == passed and summary['outcome'] == ('SEED0_GATE_MET' if passed else 'FINAL_TARGET_NOT_MET')
            and summary['multi_seed_goal_complete'] is False, 'Inconsistent seed0 outcome')
    for contrast in ('B-A', 'B-C'):
        for key in CELLS:
            require(summary['mechanism_delta_degrees'][contrast][key] == summary['metrics']['B'][key] - summary['metrics'][contrast[-1]][key], 'Changed mechanism delta')
    return summary, selection, done


class Report:
    def __init__(self):
        self.md, self.sections, self.toc = [], [], []
    def section(self, title):
        number = len(self.toc) + 1
        self.toc.append(title)
        self.md.append(f'\n## {number:02d} {title}\n')
        self.sections.append(f'<section class="sec" id="s{number}"><div class="sec-head"><span class="sec-num">{number:02d}</span><h2>{html.escape(title)}</h2></div>')
    def paragraph(self, text):
        public_safe(text)
        self.md.append(text + '\n')
        self.sections.append('<p>' + html.escape(text) + '</p>')
    def table(self, headers, rows):
        rows = [[public_safe(str(x)) for x in row] for row in rows]
        self.md += ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        self.md += ['| ' + ' | '.join(x.replace('|', '\\|').replace('\n', ' ') for x in row) + ' |' for row in rows]
        self.md.append('')
        self.sections.append('<div class="tbl-wrap"><table class="responsive-table"><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td data-label="{html.escape(h, quote=True)}">{html.escape(v)}</td>' for h, v in zip(headers, row)) + '</tr>' for row in rows) + '</tbody></table></div>')
    def link(self, label, path):
        self.md.append(f'[{label}]({path})\n')
        self.sections.append(f'<p><a href="{html.escape(path, quote=True)}">{html.escape(label)}</a></p>')


def render(root):
    summary, selection, done = load(root)
    r = Report()
    r.section('判定与边界')
    r.paragraph(f"冻结开发预选：{selection['winner']}；最终判定：{summary['outcome']}。单 seed0 闸门通过：{summary['winner_passes_seed0_gate']}；多 seed 总目标未完成。")
    r.paragraph('通过条件是预选的新候选在全部18格严格低于原baseline；任何一格未通过均不能称总体成功。即使seed0通过，也不代表统计显著、完成多seed目标或证明新颖性。本批评分后结束，不按test细节改参数。历史test曾被查看，见USER_TASK。')
    r.section('对照与资源边界')
    r.table(['方案', '定义', '解释'], [('baseline', '本项目冻结的modified One-Net baseline', '固定性能参照；非重新复现作者完整协议'), ('O', '既有数据组成父项', '不新增本轮外部先验的既有参照'), ('A', 'W × H', '保留旧明暗，叠加新照明'), ('B', '(W / s0) × H', '与A使用相同H及色度GT'), ('C', '(W / s0) × rot180(H)', '重新计算图像、GT和中性目标')])
    r.paragraph('B-A检验有限度去旧明暗；B-C检验新照明与场景的空间对应。B-O包含外部先验及有效性变化，不能全部归因于去旧明暗。三臂保留相同技术有效性和父项未加权patch标签。')
    r.section('冻结开发五项与预选')
    r.paragraph('五项误差单位为度，越低越好；开发score无量纲，沿冻结公式。只按开发score选型，同分依次O、A、B、C，最终test不参与改选。')
    r.table(['子项', 'O', 'A', 'B', 'C', 'B-A', 'B-C'], [(mode, *(selection['results'][a]['means'][mode] for a in MODELS[1:]), selection['results']['B']['means'][mode] - selection['results']['A']['means'][mode], selection['results']['B']['means'][mode] - selection['results']['C']['means'][mode]) for mode in MODES])
    r.table(['候选', 'score', '预选'], [(a, selection['scores'][a], a == selection['winner']) for a in MODELS[1:]])
    r.section('完整18格最终误差')
    r.paragraph('单位：度，越低越好。patch pooled汇总全部有效patch；patch image-balanced先算每图patch均值再等权平均；pixel image-balanced先算每图pixel均值再等权平均。all为全部图，single为单光源，multi为多光源。所有方案沿原评价身份和测试mask；这些误差依赖评价GT，真实推理不能获得。数值保留JSON原精度。')
    r.table(['split / group / metric', *MODELS], [(key, *(summary['metrics'][a][key] for a in MODELS)) for key in CELLS])
    r.section('机制差值与局限')
    r.paragraph('差值单位：度。B-A或B-C小于0表示B误差更低，大于0表示B更高；仅为单seed描述性观察。相同场景的多个composition不是独立样本，场景bootstrap亦不能代表训练seed不确定性。没有干预强度证据时，不宣称空间对应机制得到充分验证。')
    for contrast in ('B-A', 'B-C'):
        values = summary['mechanism_delta_degrees'][contrast]
        lower = sum(values[k] < 0 for k in CELLS)
        r.paragraph(f"{contrast}：B在18格中的{lower}格误差更低；test/all/patch_pooled差值为{values['test/all/patch_pooled']}度。这是描述性差异，不是统计显著性结论。")
    r.table(['split / group / metric', 'B-A', 'B-C'], [(k, summary['mechanism_delta_degrees']['B-A'][k], summary['mechanism_delta_degrees']['B-C'][k]) for k in CELLS])
    for name in ('per_image_deltas.csv', 'per_scene_deltas.csv'):
        r.link(name, 'results/' + name)
    r.section('离线成本、训练成本与校准')
    cache = read(root / 'cache_manifest.json')
    cache_progress = read(root / 'cache_progress.json')
    preflight = read(root / 'preflight_001.json')
    r.table(['阶段', '耗时秒', '峰值allocated bytes', '其他'], [('全来源缓存', cache['seconds'], cache_progress['peak_gpu_bytes'], f"{len(cache['files'])}来源；{cache['bytes']} bytes；不可用比例{cache['unusable_source_fraction']}"), ('渲染preflight', preflight['seconds'], preflight['peak_allocated_bytes'], preflight['scope']), *((a + '正式训练', done[a]['seconds'], done[a]['peak_gpu_bytes'], f"{done[a]['steps']}成功更新；{done[a]['cycles']}cycles") for a in 'ABC')])
    rows = []
    for arm in 'ABC':
        calibration = read(root / f'calibration/attempt_001/{arm}/completion.json')
        rows.append((arm, calibration['status'], calibration['steps'], calibration['seconds'], calibration['peak_gpu_bytes']))
    r.table(['校准臂', '状态', '更新数', '耗时秒', '峰值allocated bytes'], rows)
    r.paragraph('校准独立于正式初始化与预算；preflight不是optimizer或性能证据。峰值allocated不等于整卡占用。训练耗时含本轮在线构造，不能将历史耗时作为本次全流程耗时。')
    online, consumption = [], []
    for arm in 'ABC':
        records = [json.loads(line) for line in (root / f'runs/{arm}_seed0/history.jsonl').read_text().splitlines() if line.strip()]
        times = [x['online_seconds_per_update'] for x in records]
        require(times and all(finite(t) and t > 0 for t in times), 'Missing online timing evidence')
        weights = [len(x['updates']) for x in records]
        require(all(n > 0 for n in weights) and sum(weights) == done[arm]['steps'], 'Incomplete online timing coverage')
        mean = sum(t * n for t, n in zip(times, weights)) / sum(weights)
        online.append((arm, mean, min(times), max(times)))
        totals = {key: 0 for key in ('consumed_views', 'valid_patches', 'consumed_mixed_views', 'consumed_mixed_valid_patches')}
        for record in records:
            for key in totals:
                owner = record['source_render_diag'] if key.startswith('consumed_mixed_') else record
                value = owner[key]  # Missing evidence is an error, never inferred.
                require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                        f'Invalid actual consumption field: {arm}/{key}')
                totals[key] += value
        consumption.append((arm, len(cache['files']), done[arm]['cycles'] * 2 * len(cache['files']),
                            totals['consumed_views'], totals['valid_patches'],
                            totals['consumed_mixed_views'], totals['consumed_mixed_valid_patches']))
    r.table(['训练臂', '在线加权均值秒/update', '在线最小秒/update', '在线最大秒/update'], online)
    r.paragraph('在线均值按每cycle实际更新数加权；范围为各cycle的秒/update。包含该cycle构造与更新时间，不包含全流程离线缓存与所有保存开销。')
    r.table(['训练臂', '来源图数', '生成视图slots', '实际消费视图slots', '实际有效patch slots', '实际混光视图slots', '实际混光有效patch slots'], consumption)
    r.paragraph(f"来源图数为cache_manifest中的{len(cache['files'])}。每cycle每来源生成两视图，因此生成数=cycles×{2 * len(cache['files'])}；实际消费四项直接累加history字段。尾轮截断使生成数与实际消费数不同。视图slot和patch slot是重复生成/消费记录，不是独立照片或独立观测；来源图数也不能直接等同独立场景数。")
    r.section('外部先验与未验证事项')
    priors = read(root / 'priors_manifest.json')
    r.table(['记录', '证据内容'], [(k, json.dumps(priors[k], ensure_ascii=False)) for k in ('licenses', 'external_training_sources', 'nikon_overlap')])
    r.paragraph('Intrinsic灰度分解及DSINE法线为外部学习先验，不是实测albedo或法线；camera-linear proxy为域外近似，近似内参不等于准确几何。未恢复完整BRDF、镜面/透明、互反射或真实投影阴影。TIFF黑电平与转换来源仍UNVERIFIED，按REVISION_1接受工程假设。真实换光物理验证：NOT_PERFORMED。风险区域可进入整图上下文，技术mask不保证纯漫反射。')
    r.section('独立审查与可追溯证据')
    review = root / 'reviews/final_metrics_review.md'
    if review.exists():
        review_text = public_safe(review.read_text())
        r.paragraph('下列内容为独立审查文件原文；本生成器不自行授予PASS，不将报告生成视为验收通过。')
        r.md.append(review_text)
        r.sections.append('<pre>' + html.escape(review_text) + '</pre>')
        r.link('独立最终指标审查', 'reviews/final_metrics_review.md')
    else:
        r.paragraph('待独立审查：reviews/final_metrics_review.md 尚不存在。本报告不声明独立验收通过。')
    paths = ['USER_TASK.txt', 'REVISION_1.md', 'run_manifest.json', 'frozen/manifest.json', 'development_selection.json', 'results/summary.json', 'results/evaluation_manifest.json', 'cache_manifest.json', 'cache_progress.json', 'preflight_001.json', 'priors_manifest.json', 'build_report.py']
    if review.exists():
        paths.append('reviews/final_metrics_review.md')
    r.table(['证据文件', 'SHA256'], [(p, sha(root / p)) for p in paths])
    for path in paths:
        r.link(path, path)
    css = re.search(r'<style>(.*?)</style>', TEMPLATE.read_text(), re.S).group(1)
    css += '\nmain,td{min-width:0;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.82rem}table{table-layout:fixed}.tbl-wrap{overflow:visible}'
    title = 'Diffuse Virtual GT v1：seed0结果与证据边界'
    body = ''.join(r.sections)
    # Close each section before opening the next and before the footer.
    body = body.replace('<section class="sec"', '</section><section class="sec"', len(r.toc))[10:] + '</section>'
    toc = ''.join(f'<li><a href="#s{i}">{i:02d} {html.escape(t)}</a></li>' for i, t in enumerate(r.toc, 1))
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head><body><div class="wrap"><header class="hero"><div class="eyebrow">Diffuse Virtual GT · evidence report</div><h1 class="title">{title}</h1><p class="lede">冻结开发预选、统一最终评分与外部先验边界。</p><div class="rule"></div></header><div class="grid"><aside class="toc"><div class="toc-label">目录</div><ol>{toc}</ol></aside><main>{body}<footer>agent: codex · Markdown源：FINAL_REPORT.md · 只读JSON生成；数值验收以独立审查为准。</footer></main></div></div></body></html>'
    return '# ' + title + '\n\nagent: codex\n' + '\n'.join(r.md), page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate completed evidence without generating files')
    args = parser.parse_args()
    if args.check_only:
        load(ROOT)
        print('Completed evidence schema/bindings checked; independent review is separate.')
        return
    targets = [ROOT / 'FINAL_REPORT.md', ROOT / 'FINAL_REPORT.html']
    require(not any(p.exists() for p in targets), 'Existing report preserved; archive it before regenerating')
    md, page = render(ROOT)
    for target, content in zip(targets, (md, page)):
        with target.open('x') as stream:
            stream.write(content)
        print(f'{target}: sha256={sha(target)}')


if __name__ == '__main__':
    main()
