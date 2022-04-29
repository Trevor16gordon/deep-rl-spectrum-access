"""File for visualization functions
"""
import pdb
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_freq_status_over_time(data):
    """Plot a grid show frequency band status over time

    Args:
        data (np.array): shape (num_time_slots, num_freq_bands)
            values should be:
                0 for unused
                1 for used
                2 for collision
    """
    fig = px.imshow(data, color_continuous_scale=["#F0F0F0", "#77DD76", "#ff6962"])
    fig.update_layout(height=500,
                        width=500,
                        yaxis_title="Frequency Band",
                        xaxis_title="Time Slot")
    return fig

def plot_and_save_freq_status_over_time(data, filepath):
    """Save plot

    Args:
        data (np.array): shape (num_time_slots, num_freq_bands)
            values should be:
                0 for unused
                1 for used
                2 for collision
    """
    fig = plot_freq_status_over_time(data)
    fig.write_image(filepath)

def plot_and_save_freq_status_and_network_output(freq_status_input, action_value, filepath):
    """Save plot

    Args:
        data (np.array): shape (num_time_slots, num_freq_bands)
            values should be:
                0 for unused
                1 for used
                2 for collision
    """
    # Flipping so that most recent time is on the right
    freq_status_input = np.flip(freq_status_input, axis = 1)
    action_value2 = np.concatenate([2*np.ones(len(action_value)-1), action_value, 2*np.ones(1)]).reshape((-1, 1))
    action_value2_text = np.concatenate([np.array([""]*(len(action_value)-1)), np.around(action_value, decimals=3), np.array([""])]).reshape((-1, 1))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.9, 0.1])
    fig.data = []
    _ = fig.add_trace(go.Heatmap(z=freq_status_input, text=freq_status_input,  texttemplate="%{text}", colorscale=["#F0F0F0", "#77DD76", "#ff6962"], zmin=0, zmax=2), row=1, col=1)
    _ = fig.add_trace(go.Heatmap(z=action_value2, text=action_value2_text,  texttemplate="%{text}", colorscale=["#F0F0F0", "#EB8909", "#ffffff"], zmin=0, zmax=2), row=1, col=2)
    _ = fig.update_layout(height=500, width=800, title="Input frequency information over time and resulting action values", yaxis_title="Frequency Band", xaxis_title="Time Slot")
    _ = fig.update_xaxes(showticklabels=False)
    _ = fig.update_yaxes(showticklabels=False)
    _ = fig.update_traces(showscale=False)
    _ = fig.update_traces(colorbar_tickmode="array", colorbar_ticktext=[1,2,3,4,5,6])
    _ = fig.write_image(filepath)

# colorscale=["#F0F0F0", "#EB8909", "#000000"], zmin=0, zmax=2
# Color scale for probability dist is gray to orange to black
# The black is for the cells that are not included


# fig.add_trace(go.Heatmap(z=action_value, text=np.around(action_value, decimals=3).astype(str),
# fig.add_trace(go.Heatmap(z=action_value, colorscale='BuPu'), row=1, col=2)

def plot_spectrum_usage_over_time(df, filepath=None, add_collisions=False):
    """Save plot
    Plot's 2 main plots on the same axis

    df expected to be the output of agent_actions_to_information_table
    """
    agent_name_cols = [col for col in df.columns if ("agent" in col) and (len(col) == 7)]
    agent_name_cols = sorted(agent_name_cols, key=lambda x: int(x.replace("agent_", "")))
    band_name_cols = [col for col in df.columns if "band" in col]
    rew_col_names = [col for col in df.columns if "reward_agent" in col]
    cum_rew_col_names = [col for col in df.columns if "cum_reward_agent" in col]
    cum_rew_col_names = sorted(cum_rew_col_names, key=lambda x: int(x.replace("cum_reward_agent_", "")))
    r_col_names = [*rew_col_names, *cum_rew_col_names]

    num_agents = len(agent_name_cols)

    df1 = df.melt(id_vars="time", value_vars=agent_name_cols, var_name='Agent', value_name='Action')
    df2 = df.melt(id_vars="time", value_vars=rew_col_names, var_name='Agent', value_name='Reward')
    df3 = df.melt(id_vars="time", value_vars=cum_rew_col_names, var_name='Agent', value_name='Cumulative Reward')
    df4 = df.melt(id_vars="time", value_vars=band_name_cols, var_name='Frequency', value_name='Status')

    fig = go.Figure()
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}],
                           [{"secondary_y": True}]])
    fig.data = []

    col_freq_usage = 1
    row_freq_usage = 1
    col_reward = 1
    row_reward = 2

    colors = px.colors.qualitative.Plotly


    for ii, name in enumerate(agent_name_cols):
        df_sub = df1.loc[df1["Agent"]== name, :]
        fig.add_trace(go.Scatter(x=df_sub["time"], y=df_sub["Action"], fill='tonexty',
                            mode='none',
                            name=name,
                            fillcolor=colors[ii]
                            ),
                            row=row_freq_usage,
                            col=col_freq_usage)

    
    if add_collisions:
        x_collsions, y_collisions = (df[band_name_cols] == 2).values.nonzero()
        y_collisions += 1
        fig.add_trace(go.Scatter(x=x_collsions, y=y_collisions,
                                mode='markers',
                                name=f"collisions",
                                marker=dict(size=10, color="red", symbol="x")
                                ),
                            row=row_freq_usage,
                            col=col_freq_usage)

    fig.update_layout(plot_bgcolor='rgb(250,250,250)')

    collis_cols = ["rgb(119, 221, 118, 0.025)", "rgb(240, 240, 240, 0.025)", "rgb(255, 105, 98, 0.025)"]
    for ii, name in enumerate(["moving_throughput", "moving_no_transmit", "moving_collision_dens"]):
        df_sub = df3.loc[df3["Agent"]== name, :]
        fig.add_trace(go.Scatter(x=df["time"], y=df[name],
                            name=name,
                            line=dict(width=0.1, color=collis_cols[ii]),
                            stackgroup='one'),
                            secondary_y=True,
                            row=row_reward,
                            col=col_reward
                            )

    for ii, name in enumerate(cum_rew_col_names):
        df_sub = df3.loc[df3["Agent"]== name, :]
        fig.add_trace(go.Scatter(x=df_sub["time"], y=df_sub["Cumulative Reward"],
                            name=name,
                            line=dict(color=colors[ii])),
                            row=row_reward,
                            col=col_reward
                            )
    

    fig.update_layout(
        plot_bgcolor='rgb(250,250,250)',
        title="Cumulative Reward Per Agent",
        width=1000, height=600
        )
    fig['layout']['yaxis1']['title'] = "Frequency Band Status"
    fig['layout']['yaxis2']['title'] = "Cumulative Reward"
    fig['layout']['xaxis2']['title'] = "Time Step"

    if filepath:
        fig.write_image(filepath)
    return fig