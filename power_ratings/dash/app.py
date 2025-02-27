import logging
import typing as T
import re
import datetime
import json

from dash import Dash, dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import os
from io import StringIO
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go

GRAPH_TAB = "Graph View"
TABLE_TAB = "Table View"

TEAM_TRENDS = "Team Trends"
SEASON_TRENDS = "Season Trends"

logging.basicConfig()

prefixes = {
    "M": "NCAA Men",
    "W": "NCAA Women",
}
sort_col = "CombinedRating"
plottable_cols = [
    "CombinedRating",
    "WP16",
    "OffensiveRating",
    "DefensiveRating",
    "EloWithScore",
    "EloWinLoss",
    "EloDelta21Days",
    "PossessionEfficiencyFactor",
    "TempoEstimate",
]
graphs_per_row: int = 2

datasets = {}
data_dir = "/app/output_data/"
ls = os.listdir(data_dir)
try:
    with open("output_data/build_data.json", "r") as f:
        build_data = json.load(f)
        date = build_data["build_date"]
        data_data = build_data["data_date"]
except Exception as e:
    date = datetime.date(2023, 2, 17).strftime("%Y-%m-%d")
    data_data = ""
    date = e

years_set = set()


for pref in prefixes:
    filename = f"{pref}_data_complete.csv"
    try:
        path = os.path.join(data_dir, filename)
        datasets[prefixes[pref]] = pd.read_csv(
            path,  # usecols=["Season", "TeamName"] + plottable_cols
        ).sort_values(["Season", sort_col], ascending=[False, False])
        if "TeamID" in datasets[prefixes[pref]].columns:
            datasets[prefixes[pref]] = datasets[prefixes[pref]].drop(columns="TeamID")
        if "WP16" in datasets[prefixes[pref]].columns:
            datasets[prefixes[pref]]["WP16"] = np.round(
                datasets[prefixes[pref]]["WP16"].values, 2
            )
        years_set.update(datasets[prefixes[pref]].Season.unique().tolist())
    except Exception as e:
        logging.info(f"error loading dataset {filename}: {e}")

years = sorted(list(years_set))

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css", dbc.themes.GRID]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server


def get_graphs(df, graph_type: str, team: str):
    graphs = []

    if graph_type == SEASON_TRENDS:
        yearly = df.groupby("Season")
        yearly_max = yearly.max().reset_index()
        yearly_min = yearly.min().reset_index()
        yearly_med = yearly.median().reset_index()

        for col in plottable_cols:
            fig = go.Figure()
            fig.update_layout(
                title=f"{col} vs Season",
                xaxis={"title": "Season"},
                yaxis={"title": col},
            )
            fig.add_trace(
                go.Scatter(
                    x=yearly_max.Season,
                    y=yearly_max[col],
                    line=go.scatter.Line(color="gray"),
                    showlegend=True,
                    name="max",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=yearly_med.Season,
                    y=yearly_med[col],
                    line=go.scatter.Line(color="black"),
                    showlegend=True,
                    name="median",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=yearly_min.Season,
                    y=yearly_min[col],
                    line=go.scatter.Line(color="gray"),
                    showlegend=True,
                    name="min",
                )
            )
            desc = yearly.describe()[col]
            lower_quartile = desc["25%"].values.tolist()
            upper_quartile = desc["75%"].values.tolist()
            logging.info(lower_quartile)
            fig.add_trace(
                go.Scatter(
                    x=desc.index.values.tolist()
                    + desc.index.values[::-1].tolist(),  # x, then x reversed
                    y=upper_quartile
                    + lower_quartile[::-1],  # upper, then lower reversed
                    fill="toself",
                    fillcolor="rgba(60,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="interquartile",
                )
            )

            fig.update_layout(
                title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"}
            )
            graphs.append(fig)
    elif graph_type == TEAM_TRENDS:
        df_sub = df[df.TeamName == team].copy()
        if df_sub.shape[0] > 0:
            for col in plottable_cols:
                mean = df[col].mean()
                ymin_sub = df_sub[col].min()
                ymin_sub = ymin_sub - (0.1 * abs(mean))
                ymax_sub = df_sub[col].max()
                ymax_sub = ymax_sub + (0.1 * abs(mean))
                ymin = np.min([mean, ymin_sub])
                ymax = np.max([mean, ymax_sub])
                fig = px.scatter(
                    df_sub,
                    x="Season",
                    y=col,
                    title=f"{col} vs Season for {team}",
                    trendline="lowess",
                    range_y=(ymin, ymax),
                    hover_data=plottable_cols,
                )
                graphs.append(fig)
    return graphs


@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("mw-dropdown", "value"),
    prevent_initial_call=True,
)
def func(n_clicks, value):
    df = datasets[value]
    f = StringIO()
    df.to_csv(f, index=False)
    f.seek(0)
    return dict(
        content=f.read(), filename=f"{value.replace(' ', '_')}_jamesmf_ratings.csv"
    )


# callback to make the graph-specific input row hidden/shown
@app.callback(
    Output("graph-inputs-row", "style"),
    Input("tabs-id", "value"),
)
def hide_show_graph_specific_row(tab_value):
    if tab_value != GRAPH_TAB:
        return {"display": "none"}
    return {}


# callback to make the team-graph-specific input hidden/shown
@app.callback(
    Output("team-dropdown-div", "style"),
    Input("graph-type-dropdown", "value"),
)
def hide_show_team_specific_input(graph_type_value):
    if graph_type_value != TEAM_TRENDS:
        return {"display": "none"}
    return {}


# callback to make the team-graph-specific input hidden/shown
@app.callback(
    Output("team-dropdown", "options"),
    Input("mw-dropdown", "value"),
)
def update_teamname_dropdown(mw_value):
    return datasets[mw_value].TeamName.values.tolist()


# styles
tabs_styles = {"height": "50%"}

graph_input_row = html.Div(
    dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    options=[SEASON_TRENDS, TEAM_TRENDS],
                    value=SEASON_TRENDS,
                    id="graph-type-dropdown",
                    clearable=False,
                    searchable=False,
                ),
                width={"offset": 0},
                md=2,
            ),
            dbc.Col(
                html.Div(
                    dcc.Dropdown(
                        options=[""],
                        value="",
                        id="team-dropdown",
                        placeholder="Select a team",
                    ),
                    id="team-dropdown-div",
                ),
                md=2,
            ),
        ],
        justify="center",
    ),
    id="graph-inputs-row",
    style={"display": "none"},
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dbc.Container(
            [
                dbc.Row(
                    html.H2("jamesmf power ratings", style={"text-align": "center"}),
                ),
                dbc.Row(
                    dcc.Markdown(f"Last Updated: {date} {data_data}"),
                    style={"text-align": "center"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Link(
                                    "Code",
                                    href="https://github.com/jamesmf/ncaa_bball_ratings",
                                    target="_blank",
                                ),
                                " - ",
                                dcc.Link(
                                    "Data",
                                    href="https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data",
                                    target="_blank",
                                ),
                                " - ",
                                dcc.Link(
                                    "About",
                                    href="https://github.com/jamesmf/ncaa_bball_ratings/blob/main/extra_md/definitions.md",
                                    target="_blank",
                                ),
                            ],
                        ),
                    ],
                    style={"text-align": "center"},
                ),
                dbc.Row([html.Div("", style={"height": "10px"})]),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.RangeSlider(
                                np.min(years),
                                np.max(years),
                                id="year-dropdown",
                                marks={i: f"{i}" for i in (years[0], years[-1])},
                                value=[np.max(years) - 0.01, np.max(years)],
                                step=1.0,
                                dots=False,
                                tooltip={"placement": "top", "always_visible": False},
                            ),
                            md=4,
                            sm=8,
                        )
                    ],
                    align="top",
                    justify="center",
                ),
                dbc.Row([html.Div("", style={"height": "10px"})]),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                dcc.Dropdown(
                                    [TABLE_TAB, GRAPH_TAB],
                                    TABLE_TAB,
                                    id="tabs-id",
                                    searchable=False,
                                    clearable=False,
                                ),
                            ),
                            md=2,
                            sm=4,
                            width={"offset": 0},
                            style=tabs_styles,
                        ),
                        dbc.Col(
                            html.Div(
                                dcc.Dropdown(
                                    list(datasets.keys()),
                                    "NCAA Women",
                                    id="mw-dropdown",
                                    searchable=False,
                                    clearable=False,
                                ),
                                style={"margin": "auto"},
                            ),
                            md=2,
                            sm=4,
                            width={"offset": 0},
                        ),
                    ],
                    align="top",
                    justify="center",
                ),
                dbc.Row([html.Div("", style={"height": "10px"})]),
                graph_input_row,
                dbc.Row([html.Div("", style={"height": "10px"})]),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                dbc.Button(
                                    "Download Data",
                                    id="btn-download-csv",
                                ),
                            ),
                            md=1,
                            width={"offset": 0},
                        ),
                    ],
                    justify="center",
                ),
                dbc.Row([html.Div("", style={"height": "10px"})]),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Download(id="download-csv"),
                            ],
                            md=4,
                            width={"offset": 2},
                        ),
                    ],
                ),
                dbc.Row([html.Div("", style={"height": "10px"})]),
                dbc.Row(
                    [dbc.Col(id="tabs-content", md=8, sm=12)],
                    justify="center",
                ),
            ],
            fluid=True,
        ),
    ]
)

cell_style = {
    "padding": "1px",
    "fontSize": "12px",
    "minWidth": "30px",
    "width": "60px",
    "maxWidth": "300px",
    "whiteSpace": "normal",
}


def camel_split(x: str) -> str:
    return re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", x)


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs-id", "value"),
    Input("mw-dropdown", "value"),
    Input("graph-type-dropdown", "value"),
    Input("team-dropdown", "value"),
    Input("year-dropdown", "value"),
)
def display_value(tab_value, mw_value, graph_type_value, team_value, year_value):
    df = datasets[mw_value]

    if tab_value == GRAPH_TAB:
        graphs = get_graphs(df, graph_type_value, team_value)
        rows = []
        for ind in range(0, len(graphs), graphs_per_row):
            width = int(99 / graphs_per_row)
            # style = {"maxWidth": f"500"}
            style = {}
            row = dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(f"graphs-{i + ind}", figure=g, style=style),
                        md=5,
                        sm=12,
                    )
                    for i, g in enumerate(graphs[ind : ind + graphs_per_row])
                ],
                align="center",
                justify="center",
            )
            rows.append(row)
        return html.Div(
            id="graph-content",
            children=dbc.Container(
                # [dcc.Graph(f"graphs-{i}", figure=g) for i, g in enumerate(graphs)]
                rows
            ),
        )
    else:
        df = df[df.Season.between(year_value[0] - 0.01, year_value[1] + 0.01)]
        datatable = (
            dash_table.DataTable(
                df.to_dict("records"),
                [{"name": camel_split(i), "id": i} for i in df.columns],
                style_table={
                    "overflowX": "auto",
                    "minWidth": "100%",
                    "justify": "center",
                },
                style_header={
                    "backgroundColor": "rgb(210, 210, 210)",
                    "color": "black",
                    "height": "60px",
                    "maxWidth": "50px",
                    "minWidth": "30px",
                    "whiteSpace": "normal",
                },
                fixed_columns={"headers": True, "data": 2},
                style_cell={
                    "maxWidth": "50px",
                    "minWidth": "30px",
                },
                style_data={
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": "{EloDelta21Days} > 0",
                            "column_id": "EloDelta21Days",
                        },
                        "color": "darkgreen",
                    },
                    {
                        "if": {
                            "filter_query": "{EloDelta21Days} < 0",
                            "column_id": "EloDelta21Days",
                        },
                        "color": "darkred",
                    },
                ]
                # filter_action="native",
                # sort_action="native",
            ),
        )

        return datatable


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0")
