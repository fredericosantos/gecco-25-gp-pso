import pandas as pd
from src.pso import SwarmBatch
from src.benchmarks import BenchmarkFunction, PARTITIONS
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np


def create_animation(
    swarm: SwarmBatch,
    benchmark,
    grid,
    renderer: str | None = None,
    transition_duration=300,
):
    min_bounds = min(swarm.bounds[0], benchmark.optimum_bounds[0])
    max_bounds = max(swarm.bounds[1], benchmark.optimum_bounds[1])

    optimum = benchmark.optimum.cpu().numpy()
    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=pos[0, :, 0],
                    y=pos[0, :, 1],
                    mode="markers",
                    marker=dict(size=5, color="white"),
                    opacity=0.8,
                )
            ],
            name=f"Frame {i}",
        )
        for i, pos in enumerate(swarm.history)
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=swarm.history[0][0, :, 0],
                y=swarm.history[0][0, :, 1],
                mode="markers",
                marker=dict(size=5, color="white"),
                opacity=0.8,
                showlegend=False,
            ),
            go.Scatter(
                x=[optimum[0, 0]],
                y=[optimum[0, 1]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="star"),
                opacity=1.0,
                showlegend=False,
            ),
            go.Heatmap(
                z=grid,
                x=np.linspace(min_bounds, max_bounds, grid.shape[1]),
                y=np.linspace(min_bounds, max_bounds, grid.shape[0]),
                colorscale="Viridis",
                showscale=False,
            ),
        ],
        layout=go.Layout(
            template="plotly_dark",
            title="PSO Swarm Animation",
            transition={
                "duration": transition_duration,
                "easing": "quad-in-out",
            },
            height=800,
            width=800,
            xaxis=dict(range=[min_bounds, max_bounds]),
            yaxis=dict(range=[min_bounds, max_bounds]),
            showlegend=False,
            sliders=get_sliders(swarm, transition_duration),
            updatemenus=get_updatemenus(transition_duration),
        ),
        frames=frames,
    )

    if renderer is not None:
        fig.show(renderer=renderer)
    return fig


def create_heatmap_animation(
    swarm: SwarmBatch,
    benchmark: BenchmarkFunction,
    grid_size=100,
    renderer=None,
    transition_duration=300,
):
    min_bounds = min(swarm.bounds[0], benchmark.optimum_bounds[0])
    max_bounds = max(swarm.bounds[1], benchmark.optimum_bounds[1])
    optimum = benchmark.optimum.cpu().numpy()
    # assert dimensions = 2
    assert swarm.history[0].shape[2] == 2, (
        "Only 2D swarms are supported for heatmap animations"
    )
    x_bins = np.linspace(min_bounds, max_bounds, grid_size)
    y_bins = np.linspace(min_bounds, max_bounds, grid_size)
    frames = []

    for i, positions in enumerate(swarm.history):
        heatmap, _, _ = np.histogram2d(
            positions[0, :, 0], positions[0, :, 1], bins=[x_bins, y_bins]
        )
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(z=heatmap.T, x=x_bins, y=y_bins, colorscale="Viridis")
                ],
                name=f"Frame {i}",
            )
        )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.histogram2d(
                    swarm.history[0][0, :, 0],
                    swarm.history[0][0, :, 1],
                    bins=[x_bins, y_bins],
                )[0].T,
                x=x_bins,
                y=y_bins,
                colorscale="Viridis",
            ),
            go.Scatter(
                x=[optimum[0, 0]],
                y=[optimum[0, 1]],
                mode="markers",
                marker=dict(size=10, color="red", symbol="star"),
                opacity=1.0,
            ),
        ],
        layout=go.Layout(
            template="plotly_dark",
            title="PSO Swarm Animation",
            transition={
                "duration": transition_duration,
                "easing": "quad-in-out",
            },
            height=800,
            width=800,
            xaxis=dict(range=[min_bounds, max_bounds]),
            yaxis=dict(range=[min_bounds, max_bounds]),
            showlegend=False,
            sliders=get_sliders(swarm, transition_duration),
            updatemenus=get_updatemenus(transition_duration),
        ),
        frames=frames,
    )
    fig.show(renderer=renderer)


def get_updatemenus(transition_duration: int):
    return [
        {
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": transition_duration,
                                "redraw": False,
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": transition_duration,
                                "easing": "quad-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
        }
    ]


def get_sliders(swarm: SwarmBatch, transition_duration: int):
    return [
        {
            "pad": {"b": 10},
            "x": 0.25,
            "y": 0,
            "len": 0.75,
            "steps": [
                {
                    "args": [
                        [f"Frame {i}"],
                        {
                            "frame": {
                                "duration": transition_duration,
                                "redraw": False,
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": transition_duration,
                                "easing": "quad-in-out",
                            },
                        },
                    ],
                    "label": str(i),
                    "method": "animate",
                }
                for i in range(len(swarm.history))
            ],
        }
    ]


def boxplot(
    path: Path,
    df: pd.DataFrame,
    dimension: int,
    partition: int,
    cols_to_remove: list | None = None,
    save_html: bool = False,
    filename: str | None = None,
    hover_name: str | None = None,
):
    if filename is None:
        filename = f"p{partition}_fitnesses"
    if cols_to_remove is not None:
        df = df.drop(columns=cols_to_remove)
    if hover_name is not None:
        df_melt = df.melt(
            id_vars=[hover_name], var_name="Function", value_name="Fitness"
        )
    else:
        df_melt = df.melt(var_name="Function", value_name="Fitness")
    fig = px.box(
        df_melt,
        x="Function",
        y="Fitness",
        title=f"Fitness for partition {partition} (dim={dimension}) — {PARTITIONS[partition]['test']}",
        points="all",  # Show all individual points
        boxmode="group",
        color="Function",
        height=600,
        width=1000,
        template="plotly_dark",
        hover_data=hover_name,
    )
    # make title very small
    fig.update_layout(title_font_size=10)
    # hide legend
    fig.update_layout(showlegend=False, autosize=True)
    # remove x axis title
    fig.update_xaxes(title_text="")
    # make box contour line width=1
    fig.update_traces(marker=dict(size=3), line=dict(width=1))
    fig.write_image(path / f"{filename}.png", scale=2)
    if save_html:
        # save as html
        fig.write_html(path / f"{filename}.html")

    return fig
