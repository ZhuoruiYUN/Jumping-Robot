"""Publish one immutable TensorBoard step per selected frozen development result."""
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.record_writer import RecordWriter

from .precision_continuous import BASE, HOME, START
from .precision_trials import read


def selected_report(folder, checkpoint):
    for path in sorted(folder.glob('development*/manifest.json')):
        manifest=read(path)
        if (manifest.get('status')=='complete' and manifest.get('checkpoint')==str(checkpoint)
                and manifest.get('optimizer_updates')==0 and manifest.get('seed')==7001
                and manifest.get('evaluation_scope')=='precision10_v1_development'):
            summary=path.with_name('summary.json')
            return read(summary),summary.stat().st_mtime
    raise ValueError(f'Missing frozen development result for {checkpoint} in {folder}')


def publish(folder, step, decision=None):
    target=HOME/'qualification'/f'events.out.tfevents.qualification.{step:08d}'
    if target.exists():
        return
    checkpoint=decision['checkpoint'] if decision else START
    summary,wall=selected_report(folder,checkpoint)
    result=summary['precision10']
    groups=result['groups']
    if decision and (groups!=decision['groups'] or result['deaths']!=decision['deaths']):
        raise ValueError(f'Round {step} decision and selected evaluation disagree')
    scalars={
        'Qualification/omni_clean_of_64':groups['omni']['clean'],
        'Qualification/straight_clean_of_64':groups['straight']['clean'],
        'Qualification/omni_prefix_of_640':groups['omni']['prefix'],
        'Qualification/straight_prefix_of_640':groups['straight']['prefix'],
        'Qualification/physical_deaths_of_128':result['deaths'],
        'Qualification/hold_peak_deg':result['flight_plane']['hold']['lean_peak_deg'],
        'Qualification/travel_cross_peak_deg':result['flight_plane']['travel']['lean_peak_deg'],
        'Qualification/travel_yaw_peak_deg':summary['rebound_diagnostics']['moving_heading_peak_deg'],
    }
    values=[Summary.Value(tag=tag,simple_value=value) for tag,value in scalars.items()]
    target.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=target.parent,prefix='.pending-',delete=False) as stream:
        pending=Path(stream.name)
        try:
            writer=RecordWriter(stream)
            writer.write(Event(wall_time=wall,file_version='brain.Event:2').SerializeToString())
            writer.write(Event(wall_time=wall,step=step,summary=Summary(value=values)).SerializeToString())
            writer.flush()
            os.replace(pending,target)
        finally:
            pending.unlink(missing_ok=True)


def sync():
    publish(BASE,0)
    for path in sorted(HOME.glob('round_*/decision.json')):
        publish(path.parent,int(path.parent.name.split('_')[1]),read(path))


if __name__=='__main__':
    sync()
