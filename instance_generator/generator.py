#!/usr/bin/env python3
# THIS SCRIPT GENERATES THE INSTANCE:
"""
10 × 10 grid graph with integer-labeled vertices (0‒143) using NetworkX.
"""

import random
import math

import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def draw_graph(G):

    pos = nx.spring_layout(G, seed=11904657, iterations=5000, k=1)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u];  x1, y1 = pos[v]
        edge_x += [x0, x1, None];  edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.5, color="#999"), hoverinfo="none", showlegend=False)

    node_x, node_y, text = [], [], []
    colors = []

    for n in G.nodes():
        x, y = pos[n];  node_x.append(x);  node_y.append(y);  text.append(n)
        colors.append("red" if n == "LOVVFIR" else "#1f77b4")

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
                            text=text, textposition="top center",
                            textfont=dict(size=5),
                            marker=dict(size=8, color=colors),
                            hoverinfo="text", showlegend=False)

    fig = go.Figure([edge_trace, node_trace],
                    layout=go.Layout(margin=dict(l=0,r=0,t=0,b=0),
                                     xaxis=dict(visible=False),
                                     yaxis=dict(visible=False)))

    fig.write_image("grid_graph.pdf")   

def generate_grid():
    # --- parameters -------------------------------------------------------------
    width  = 11      # number of columns
    height = 11      # number of rows
    # ----------------------------------------------------------------------------

    # Create a 2-D grid with (row, col) node labels
    G = nx.grid_2d_graph(height, width)         

    # Map each (row, col) to a unique integer: row-major order
    mapping = {(r, c): r * width + c             
               for r in range(height)
               for c in range(width)}

    # Relabel the graph in-place (returns a new graph)
    G = nx.relabel_nodes(G, mapping)

    # Add airport nodes 
    airport_nodes = []
    next_id = width * height           
    step = 2

    for r in range(0, height, step):
        for c in range(0, width, step):
            anchor = mapping[(r, c)]
            G.add_node(next_id)
            G.add_edge(anchor, next_id)  # single attachment edge
            airport_nodes.append(next_id)
            next_id += 1


    return G, airport_nodes

def sample_start_time(mu=10, sigma=4, low=0, high=24):
    """Draw an integer hour from a normal distribution, truncated to [low, high)."""
    while True:
        t = np.random.normal(mu, sigma)
        if low <= t < high:
            return int(round(t))            # keep whole hours only

if __name__ == "__main__":

    G, airport_nodes = generate_grid()
    pd.DataFrame(G.edges(), columns=["source", "target"]).to_csv("edges.csv", index=False)

    draw_graph(G)

    for flights in range(30000,30001,100):

        flight_list = []

        number_flights_added = 0

        while number_flights_added < flights:
            
            flight = number_flights_added

            # 1. choose distinct source & target
            #src, tgt = random.sample(list(G.nodes()), 2)
            src, tgt = random.sample(list(airport_nodes), 2)

            # 2. shortest path
            path = nx.shortest_path(G, src, tgt)     # list of vertices

            # 3. start time (integer hour 0‒23)
            start_t = sample_start_time()

            # 4. associate each vertex with its planned time
            tmp_flight_list = []
            arrived_at_destination = False

            for hop, vertex in enumerate(path):      
                t = start_t + hop                    # +1 h per edge
                if t >= 24:                          # stop after (including) midnight
                    break

                if vertex == tgt:
                    arrived_at_destination = True
                
                tmp_flight_list.append((flight,vertex,t))

            if arrived_at_destination is True:
                flight_list = flight_list + tmp_flight_list
                number_flights_added += 1

        pd.DataFrame(flight_list, columns=["Flight_ID", "Position", "Time"]).to_csv(f"instance_{flights}.csv", index=False)

        #capacity = math.ceil(flights/len(list(G.nodes())))
        non_airport_nodes = set(G.nodes()).difference(set(airport_nodes))
        non_airport_capacity = 400
        airport_capacity = 10000
        capacities = [(vertex,non_airport_capacity) for vertex in non_airport_nodes] + [(vertex,airport_capacity) for vertex in airport_nodes]
        pd.DataFrame(capacities, columns=["Sector_ID", "Capacity"]).to_csv(f"capacity_{flights}.csv", index=False)

        pd.DataFrame(airport_nodes, columns=["Airport_Vertex"]).to_csv("airport_vertices.csv",index=False)
        

