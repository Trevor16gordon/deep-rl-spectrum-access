

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import argparse
import pdb
import numpy as np

from utils import complete_df_to_stacked


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to the saved csv", type=str)
    parser.add_argument("--visible_timesteps", "-vt", default=30,
                        help="Number of timesteps visible on x axis at a time", type=int)
    parser.add_argument("--port", "-po", default=8051,
                        help="Number of timesteps visible on x axis at a time", type=int)
    parser.add_argument("--width", "-w", default=1000,
                        help="Width of the plot", type=int)
    parser.add_argument("--height", "-he", default=800,
                        help="Height of the plot", type=int)
    args = parser.parse_args()
    merged = pd.read_csv(args.path)

    agent_name_cols = [col for col in merged.columns if ("agent" in col) and (len(col) == 7)]
    agent_name_cols = sorted(agent_name_cols, key=lambda x: int(x.replace("agent_", "")))
    cum_rew_col_names = [col for col in merged.columns if "cum_reward_agent" in col]
    cum_rew_col_names = sorted(cum_rew_col_names, key=lambda x: int(x.replace("cum_reward_agent_", "")))

    merged3 = merged.melt(id_vars="time", value_vars=cum_rew_col_names, var_name='Agent', value_name='Cumulative Reward')

    df = complete_df_to_stacked(merged)

    num_agents = df["agent"].max() + 1
    num_actions = df["action"].max() + 1

    app = Dash(__name__)

    delta_time = args.visible_timesteps

    app.layout = html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            df["time"].min(),
            df["time"].max()-delta_time,
            step=None,
            value=df["time"].min(),
            marks={str(year): str(year) for year in df["time"].unique()},
            id='time-slider'
        )
    ],
        style={"width": "70%"})

    @app.callback(
        Output('graph-with-slider', 'figure'),
        Input('time-slider', 'value'))
    def update_figure(selected_min_timestep):

        filtered_df = df[df.time > selected_min_timestep]
        filtered_df = filtered_df[filtered_df.time < selected_min_timestep + delta_time]

        filtered_merged = merged[merged.time > selected_min_timestep]
        filtered_merged = filtered_merged[filtered_merged.time < selected_min_timestep + delta_time]
        
        timesteps = filtered_df["time"].max() + 1
        values = filtered_df.loc[:, ["time", "agent", "action", "value"]
                                 ]["value"].values.reshape((-1, num_agents*num_actions)).T
        chosen_action = filtered_df.loc[:, ["time", "agent", "action", "chosen_action"]
                                        ]["chosen_action"].values.reshape((-1, num_agents*num_actions)).T.astype(int)

        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        fig.data = []

        x_names = [x for x in range(
            selected_min_timestep, selected_min_timestep+delta_time)]
        y_names = [f"agent_{i}_action_{j}" for i in range(
            num_agents) for j in range(num_actions)]

        # TODO: Update agent chosen actions with gray / green / or red!!
        arr = filtered_merged[agent_name_cols].values
        freq_status = np.zeros_like(arr)
        for i in range(num_agents):
            other_choices = np.delete(arr, i, axis=1)
            these_choices = arr[:, i].reshape((-1, 1))
            same_choice = (np.any(these_choices == other_choices, axis=1)).reshape((-1, 1))
            collisions = 2*((these_choices > 0) & same_choice)
            # collisions = 2*((these_choices > 0) &  np.isin(these_choices, other_choices))
            success = 1*((these_choices > 0) &  np.logical_not(collisions))
            # Add 1 so that after multiplying there is a difference between chosen no transmit and background cell
            # 1 is no transmit
            # 2 is success
            # 3 is collions
            freq_status[:, i] = 1+ collisions.reshape(-1) + success.reshape(-1)
        

        # filtered_merged["agent_0"].values.reshape((-1, 1)), filtered_merged[agent_name_cols].values.shape
        chosen_action_colors = np.repeat(freq_status, num_actions, axis=1).T
        chosen_action *= chosen_action_colors
        fig.add_trace(go.Heatmap(z=chosen_action, colorscale=["#F8F8F8", "#E6E6E6", "#77DD76", "#ff6962"], zmin=0, zmax=3, y=y_names, x=x_names), col=1, row=2)
        fig.add_trace(go.Heatmap(z=values, colorscale="Blues",
                                 x=x_names,
                                 y=y_names), col=1, row=3)



        # START TRYING TO GET OTHER PLOT WORKING
        max_reward =  merged3["Cumulative Reward"].max()
   

        colors = px.colors.qualitative.Plotly
        collis_cols = ["rgb(119, 221, 118, 0.025)", "rgb(240, 240, 240, 0.025)", "rgb(255, 105, 98, 0.025)"]
        for ii, name in enumerate(["moving_throughput", "moving_no_transmit", "moving_collision_dens"]):
            df_sub = merged3.loc[merged3["Agent"]== name, :]
            fig.add_trace(go.Scatter(x=merged["time"], y=merged[name],
                                name=name,
                                line=dict(width=0.1, color=collis_cols[ii]),
                                stackgroup='one'),
                                secondary_y=True,
                                row=1,
                                col=1
                                )
        high_x = [selected_min_timestep, selected_min_timestep,  selected_min_timestep+delta_time, selected_min_timestep+delta_time]
        high_y = [0, 1, 1, 0]
        fig.add_trace(go.Scatter(x=high_x, y=high_y, 
                        fill="toself", 
                        fillcolor="rgb(200, 200, 200, 0.025)",
                        opacity=0.4,
                        line=dict(width=0.1, color="rgb(200, 200, 200, 0.025)")),
                        row=1,
                        col=1,
                        secondary_y=True,
                        )

        for ii, name in enumerate(cum_rew_col_names):
            df_sub = merged3.loc[merged3["Agent"]== name, :]
            fig.add_trace(go.Scatter(x=df_sub["time"], y=df_sub["Cumulative Reward"],
                                name=name,
                                line=dict(color=colors[ii])),
                                row=1,
                                col=1
                                )
        
        

        fig.update_layout(
            plot_bgcolor='rgb(250,250,250)',
            title="RL Frequency Spectrum Sharing Agent Results",
            width=args.width, height=args.height
        )
        fig['layout']['yaxis1']['title'] = "Cumulative Reward Per Agent"
        fig['layout']['yaxis2']['title'] = "Channel Utilization"
        fig['layout']['yaxis3']['title'] = "Agent Chosen Actions"
        fig['layout']['yaxis4']['title'] = "Agent Action Values"
        fig['layout']['xaxis3']['title'] = "Time Step"

        fig.update_traces(showscale=False, row=2, col=1)
        fig.update_traces(showscale=False, row=3, col=1)

        return fig

    app.run_server(debug=True, port=args.port)
