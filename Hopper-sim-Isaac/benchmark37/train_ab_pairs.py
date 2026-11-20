"""Launch the A/B sampling comparison immediately or at the current round boundary."""
from datetime import datetime,timezone
import argparse
import fcntl
import subprocess
import time

from .precision_continuous import HOME,stage
from .precision_trials import read,save


FOCUSED=HOME/'ab_pairs/20261120_062100_541146Z'


def course(full_pairs=False,shortfall=False):
    baseline=HOME/'apex_ab1024/baseline'
    cfg=read(baseline/'course.json')
    cfg['omni']['ab_pair_training']=True
    if full_pairs or shortfall:
        cfg['omni']['ab_full_pair_focus']=True
    if shortfall:
        from .ab_preview_audit import apex_forecast_bias
        cfg['precision_pre_reversal_release']['ascent_forecast_bias_m']=apex_forecast_bias(FOCUSED/'continuous60')
    return cfg


def run_trial(folder):
    config=folder/'course.json'
    settings=read(config)
    train=stage(folder,'train','train','--headless','--ring-config',config,'--profile','vram_1024')
    checkpoint=train/'model_49.pt'
    if read(train/'manifest.json')['num_envs']!=1024 or not checkpoint.is_file():
        raise RuntimeError('Incomplete 1024-env training run')
    for name,suite in [('development','development'),('continuous60','continuous')]:
        output=stage(folder,name,'eval','--headless','--checkpoint',checkpoint,'--precision-suite',suite)
        manifest=read(output/'manifest.json')
        if manifest['optimizer_updates']!=0 or manifest['checkpoint']!=str(checkpoint):
            raise RuntimeError(f'Evaluation provenance mismatch in {output}')
        baseline=HOME/'apex_ab1024/baseline'/name
        if read(output/'cases.json')['cases']!=read(baseline/'cases.json')['cases']:
            raise RuntimeError(f'Evaluation cases differ from the 1024-env baseline in {output}')
    groups=read(folder/'development/summary.json')['precision10']['groups']
    long_run=read(folder/'continuous60/summary.json')['continuous']
    result=dict(groups=groups,continuous=long_run)
    if settings['omni'].get('ab_full_pair_focus'):
        from .ab_preview_audit import paired_preview
        reference=HOME/'ab_pairs/20261120_042037_595771Z'
        cached='ascent_forecast_bias_m' in settings['precision_pre_reversal_release']
        if cached:
            reference=FOCUSED
        control=read(reference/'course.json')
        control['omni']['ab_full_pair_focus']=True
        if cached:
            control['precision_pre_reversal_release']['ascent_forecast_bias_m']=course(shortfall=True)['precision_pre_reversal_release']['ascent_forecast_bias_m']
        if control!=settings:
            raise ValueError('Full-pair comparison requires the original AB course')
        report={}
        reference_eval=reference if cached else folder/'reference'
        for label,p,ck in [('original',reference_eval,reference/'train/model_49.pt'),('focused',folder,checkpoint)]:
            if not (cached and p==reference):
                p.mkdir(exist_ok=True)
                for suite in ('ab_continue','ab_reverse'):
                    stage(p,suite,'eval','--headless','--checkpoint',ck,'--precision-suite',suite)
            report[label]=paired_preview(p,ck)
        if read(reference_eval/'ab_continue/cases.json')['cases']!=read(folder/'ab_continue/cases.json')['cases']:
            raise ValueError('Paired test cases differ across checkpoints')
        result['paired_preview']=report
    return result


def service_trial(folder):
    """Caller holds the shared GPU worker lock. Leave interrupted stages resumable."""
    if (folder/'result.json').exists() or (folder/'error.json').exists():
        return
    try:
        result=run_trial(folder)
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
        save(folder/'error.json',dict(error=str(exc)))
    else:
        save(folder/'result.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-pairs',action='store_true',help='Increase full straight and reverse pairs to 25 percent each')
    parser.add_argument('--ascent-shortfall',action='store_true',help='Full-pair sampling with gated pre-reversal shortfall feedback')
    args=parser.parse_args()
    folder=HOME/'ab_pairs'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    folder.mkdir(parents=True)
    save(folder/'course.json',course(args.full_pairs,args.ascent_shortfall))
    print(f'AB 已登记，训练和验收目录 {folder}',flush=True)
    with (HOME/'worker.lock').open('a') as lock:
        waiting=False
        while not ((folder/'result.json').exists() or (folder/'error.json').exists()):
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                if not waiting:
                    print('GPU 训练队列正在运行。AB 将在轮次边界执行，可用现有 TensorBoard 查看。',flush=True)
                    waiting=True
                time.sleep(5)
            else:
                service_trial(folder)
                break
    if (folder/'error.json').exists():
        raise RuntimeError(read(folder/'error.json')['error'])
    result=read(folder/'result.json')
    for name,group in result['groups'].items():
        long_run=result['continuous'][name]
        print(f"{name} | 十跳 {group['clean']}/64 | 第十点后首次达标 "
              f"{long_run['after_ten_hits']}/{long_run['after_ten_attempts']} | 60 秒倒地 {long_run['deaths']}")
    print(f'结果文件 {folder/"result.json"}')


if __name__=='__main__':
    main()
