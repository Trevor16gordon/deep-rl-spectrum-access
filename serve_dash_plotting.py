

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to the saved csv", type=str)
    parser.add_argument("--visible_timesteps", "-vt", default=30,
                        help="Number of timesteps visible on x axis at a time", type=int)
    parser.add_argument("--port", "-po", default=8051,
                        help="Number of timesteps visible on x axis at a time", type=int)
    args = parser.parse_args()
    df = pd.read_csv(args.path)

    num_agents = 3
    num_actions = 2 + 1

    app = Dash(__name__)

    delta_time = args.visible_timesteps

    app.layout = html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            df["timestep"].min(),
            df["timestep"].max()-delta_time,
            step=None,
            value=df["timestep"].min(),
            marks={str(year): str(year) for year in df["timestep"].unique()},
            id='timestep-slider'
        )
    ],
        style={"width": "70%"})

    @app.callback(
        Output('graph-with-slider', 'figure'),
        Input('timestep-slider', 'value'))
    def update_figure(selected_min_timestep):
        filtered_df = df[df.timestep > selected_min_timestep]
        filtered_df = filtered_df[filtered_df.timestep <
                                  selected_min_timestep + delta_time]
        
        timesteps = filtered_df["timestep"].max() + 1
        values = filtered_df.loc[:, ["timestep", "agent", "action", "value"]
                                 ]["value"].values.reshape((-1, num_agents*num_actions)).T
        chosen_action = filtered_df.loc[:, ["timestep", "agent", "action", "chosen_action"]
                                        ]["chosen_action"].values.reshape((-1, num_agents*num_actions)).T.astype(int)

        fig = make_subplots(rows=2, cols=1)
        fig.data = []

        x_names = [x for x in range(
            selected_min_timestep, selected_min_timestep+delta_time)]
        y_names = [f"agent_{i}_action_{j}" for i in range(
            num_agents) for j in range(num_actions)]

        fig.add_trace(go.Heatmap(z=chosen_action, colorscale=[
                      "#F0F0F0", "#EB8909"], zmin=0, zmax=1, y=y_names), col=1, row=1)
        fig.add_trace(go.Heatmap(z=values, colorscale="Blues",
                                 x=x_names,
                                 y=y_names), col=1, row=2)

        fig.update_layout(
            plot_bgcolor='rgb(250,250,250)',
            title="RL Frequency Spectrum Sharing Agent Results",
            width=1000, height=800
        )
        fig['layout']['yaxis1']['title'] = "Agent Chosen Actions"
        fig['layout']['yaxis2']['title'] = "Agent Action Values"
        fig['layout']['xaxis2']['title'] = "Time Step"

        return fig

    app.run_server(debug=True, port=args.port)
