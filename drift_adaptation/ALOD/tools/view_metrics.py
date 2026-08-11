"""Streamlit dashboard for ALOD train/validation metric curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from tools.common.metrics_logs import available_rounds, load_train_frame
from tools.common.metrics_scanner import RunRef, scan_runs
from tools.common.metrics_viewer import load_round_summary_frame, load_validation_frame


alt.data_transformers.disable_max_rows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--work_dir', '--run_dir', dest='work_dir', default='work_dirs')
    args, _ = parser.parse_known_args()
    return args


def _run_option(run: RunRef) -> str:
    seeds = ','.join('seed_%d' % seed for seed in run.seeds) if run.seeds else 'seed_0'
    return '%s | %s | %s | %s | %s' % (
        run.method.upper(),
        run.detector,
        run.dataset,
        run.run_id,
        seeds,
    )


def _multiselect_all(label: str, values: Sequence[str], default_all: bool = True) -> List[str]:
    options = list(values)
    default = options if default_all else []
    return st.sidebar.multiselect(label, options=options, default=default)


@st.cache_data(show_spinner=False)
def _cached_scan(work_dir: str) -> List[Dict]:
    return [run.to_dict() for run in scan_runs(Path(work_dir))]


def _runs_from_dicts(items: Iterable[Dict]) -> List[RunRef]:
    runs = []
    for item in items:
        data = dict(item)
        data['seeds'] = tuple(data.get('seeds', ()))
        runs.append(RunRef(**data))
    return runs


@st.cache_data(show_spinner=False)
def _cached_validation(run_items: Sequence[Dict]):
    return load_validation_frame(_runs_from_dicts(run_items))


@st.cache_data(show_spinner=False)
def _cached_round_summary(run_items: Sequence[Dict]):
    return load_round_summary_frame(_runs_from_dicts(run_items))


@st.cache_data(ttl=10, show_spinner=False)
def _cached_train(run_items: Sequence[Dict], seeds: Sequence[str], rounds: Sequence[int]):
    return load_train_frame(_runs_from_dicts(run_items), seeds=seeds, rounds=rounds)


def _filter_runs(runs: List[RunRef]) -> List[RunRef]:
    methods = _multiselect_all('Methods', sorted({run.method.upper() for run in runs}))
    detectors = _multiselect_all('Detectors', sorted({run.detector for run in runs}))
    datasets = _multiselect_all('Datasets', sorted({run.dataset for run in runs}))
    experiments = _multiselect_all('Experiments', sorted({run.experiment for run in runs}))

    filtered = [
        run for run in runs
        if run.method.upper() in methods
        and run.detector in detectors
        and run.dataset in datasets
        and run.experiment in experiments
    ]
    run_options = {_run_option(run): run for run in filtered}
    default_options = list(run_options.keys())
    selected_labels = st.sidebar.multiselect(
        'Runs',
        options=default_options,
        default=default_options,
    )
    return [run_options[label] for label in selected_labels]


def _validation_section(runs: List[RunRef]) -> None:
    st.subheader('Validation Curves')
    run_items = tuple(run.to_dict() for run in runs)
    df = _cached_validation(run_items)
    if df.empty:
        st.info('No validation metrics found for the selected runs.')
        return

    metric_options = sorted(df['metric'].dropna().unique().tolist())
    preferred_metrics = ('mAP', 'AP50', 'bbox_mAP', 'bbox_mAP_50')
    default_metrics = [
        metric for metric in preferred_metrics if metric in metric_options
    ] or metric_options[:1]
    selected_metrics = st.multiselect('Validation metrics', metric_options, default=default_metrics)

    show_seeds = st.checkbox('Show seed curves', value=True)
    show_mean = st.checkbox('Show mean curves', value=True)
    show_std = st.checkbox('Show mean +/- std band', value=True)
    x_axis = st.radio('Validation x-axis', ['round', 'labeled_images'], horizontal=True)

    plot_df = df[df['metric'].isin(selected_metrics)].copy()
    allowed_types = []
    if show_seeds:
        allowed_types.append('seed')
    if show_mean:
        allowed_types.append('mean')
    plot_df = plot_df[plot_df['series_type'].isin(allowed_types)]
    if plot_df.empty:
        st.info('Select at least one curve type.')
        return

    curves = sorted(plot_df['curve_label'].unique().tolist())
    default_curves = [
        curve for curve in curves
        if (' mean ' in (' ' + curve + ' '))
        or curve.endswith(' mean mAP')
        or curve.endswith(' mean AP50')
        or curve.endswith(' mean bbox_mAP')
        or curve.endswith(' mean bbox_mAP_50')
    ] or curves
    selected_curves = st.multiselect('Validation curves', curves, default=default_curves)
    plot_df = plot_df[plot_df['curve_label'].isin(selected_curves)]
    if plot_df.empty:
        st.info('No validation curves selected.')
        return

    x_title = 'Labeled images' if x_axis == 'labeled_images' else 'Round'
    base = alt.Chart(plot_df).encode(
        x=alt.X('%s:Q' % x_axis, title=x_title),
        color=alt.Color('curve_label:N', title='Curve'),
        tooltip=[
            'method:N', 'run_id:N', 'seed:N', 'metric:N',
            alt.Tooltip('round:Q', format='.0f'),
            alt.Tooltip('labeled_images:Q', format='.0f'),
            alt.Tooltip('value:Q', format='.4f'),
        ],
    )
    line = base.mark_line(point=True).encode(y=alt.Y('value:Q', title='Metric'))
    chart = line

    band_df = plot_df[
        (plot_df['series_type'] == 'mean')
        & plot_df['value_low'].notna()
        & plot_df['value_high'].notna()
    ]
    if show_std and not band_df.empty:
        band = alt.Chart(band_df).mark_area(opacity=0.12).encode(
            x=alt.X('%s:Q' % x_axis, title=x_title),
            y=alt.Y('value_low:Q', title='Metric'),
            y2='value_high:Q',
            color=alt.Color('curve_label:N', title='Curve'),
        )
        chart = band + line
    st.altair_chart(chart.interactive(), use_container_width=True)

    table_df = _cached_round_summary(run_items)
    if not table_df.empty:
        columns = [
            column for column in [
                'method', 'dataset', 'run_id', 'seed', 'round',
                'labeled_images', 'duration_min', 'mAP', 'AP50',
                'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75'
            ]
            if column in table_df.columns
        ]
        st.dataframe(
            table_df[columns].sort_values(['method', 'run_id', 'seed', 'round']),
            use_container_width=True,
            hide_index=True,
        )


def _train_section(runs: List[RunRef]) -> None:
    st.subheader('Train Curves')
    if not runs:
        st.info('No runs selected.')
        return

    run_options = {_run_option(run): run for run in runs}
    default_train_runs = list(run_options.keys())[:1]
    selected_run_labels = st.multiselect(
        'Train runs',
        options=list(run_options.keys()),
        default=default_train_runs,
    )
    selected_runs = [run_options[label] for label in selected_run_labels]
    if not selected_runs:
        st.info('Select at least one train run.')
        return

    seed_options = sorted({
        'seed_%d' % seed
        for run in selected_runs
        for seed in (run.seeds or (0,))
    })
    selected_seeds = st.multiselect('Train seeds', seed_options, default=seed_options[:1])

    round_options = sorted({
        round_index
        for run in selected_runs
        for round_index in available_rounds(run)
    })
    default_rounds = round_options[-1:] if round_options else []
    selected_rounds = st.multiselect('Train rounds', round_options, default=default_rounds)
    if not selected_seeds or not selected_rounds:
        st.info('Select seeds and rounds to load train curves.')
        return

    train_df = _cached_train(
        tuple(run.to_dict() for run in selected_runs),
        tuple(selected_seeds),
        tuple(int(round_index) for round_index in selected_rounds),
    )
    if train_df.empty:
        st.info('No train logs found for the selected seeds/rounds.')
        return

    key_options = sorted(train_df['key'].unique().tolist())
    preferred = [key for key in ('loss', 'loss_cls', 'loss_bbox') if key in key_options]
    selected_keys = st.multiselect('Train curve keys', key_options, default=preferred or key_options[:1])
    x_axis = st.radio('Train x-axis', ['local_step', 'iter', 'epoch'], horizontal=True)

    plot_df = train_df[train_df['key'].isin(selected_keys)].copy()
    curves = sorted(plot_df['curve_label'].unique().tolist())
    default_curves = curves[:20]
    selected_curves = st.multiselect('Train curves', curves, default=default_curves)
    plot_df = plot_df[plot_df['curve_label'].isin(selected_curves)]
    if plot_df.empty:
        st.info('No train curves selected.')
        return

    chart = alt.Chart(plot_df).mark_line().encode(
        x=alt.X('%s:Q' % x_axis, title=x_axis),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('curve_label:N', title='Curve'),
        tooltip=[
            'method:N', 'run_id:N', 'seed:N',
            alt.Tooltip('round:Q', format='.0f'),
            'key:N',
            alt.Tooltip('epoch:Q', format='.0f'),
            alt.Tooltip('iter:Q', format='.0f'),
            alt.Tooltip('local_step:Q', format='.0f'),
            alt.Tooltip('value:Q', format='.4f'),
        ],
    )
    st.altair_chart(chart.interactive(), use_container_width=True)


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title='ALOD Metrics', layout='wide')
    st.title('ALOD Metrics Dashboard')
    st.caption(
        'Read-only train loss/lr and validation VOC/COCO metric viewer for ALOD work_dirs.'
    )

    with st.sidebar:
        st.header('Run Selection')
        work_dir = st.text_input('work_dirs root', value=args.work_dir)
        if st.button('Rescan'):
            st.cache_data.clear()

    runs = _runs_from_dicts(_cached_scan(work_dir))
    if not runs:
        st.warning('No ALOD runs found under: %s' % work_dir)
        return

    with st.sidebar:
        st.write('%d runs discovered' % len(runs))
        selected_runs = _filter_runs(runs)
        st.write('%d runs selected' % len(selected_runs))

    if not selected_runs:
        st.info('Select at least one run in the sidebar.')
        return

    overview = pd.DataFrame([run.to_dict() for run in selected_runs])
    st.dataframe(
        overview[['method', 'detector', 'dataset', 'experiment', 'run_id', 'seeds', 'rounds', 'budget', 'status']],
        use_container_width=True,
        hide_index=True,
    )

    _validation_section(selected_runs)
    _train_section(selected_runs)


if __name__ == '__main__':
    main()
