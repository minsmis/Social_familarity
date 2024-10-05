import os

import time
import random
import warnings

import numpy as np
import plotly.graph_objects as go
import pandas as pd

import brain.Visualization.feature.social as bf


class Clusters:
    def __init__(self):
        super(Clusters, self).__init__()

        # Storage path
        self.storage_dir = ''

        # Init
        self.__init_parameters__()
        self.__init_data__()
        self.__init_metrics__()
        self.__init_reducer__()
        self.__init_visualization__()

        # Experiment performance
        self.accuracy_list = []

    def __init_parameters__(self):
        # Preprocessing parameters
        self.q = 5
        self.fps = 20
        self.time_bin = 100  # ms
        self.dim_reduction_method = 'pca'
        self.n_dim = 3

    def __init_data__(self):
        # Data
        self.raw_data = {}

        self.data = {}
        self.data_features = {}

    def __init_metrics__(self):
        self.data_cluster_centroid = {}
        self.data_cluster_radius = {}
        self.data_cluster_centroid_distance = {}
        self.data_polygon_area = 0

    def __init_reducer__(self):
        # Reducer
        self.reducer = bf.make_reducer(method=self.dim_reduction_method, dim=self.n_dim)

    def __init_visualization__(self):
        # Visualization color
        self.color_theme = ['#8faadc', '#fc9f9f', '#004812', '#FF5520']
        self.line_color = '#9b00ae'
        self.rotating = False
        self.floating_labels = []
        self.floating_labels_fontsize = 36
        self.floating_labels_colors = []
        self.show_hinge = False

    def prepare_dataset(self, trial_label: list, data_path: str, epoch_storage: str):
        # Prepare dataset for XOR

        # Check validity: Label-Data match
        if len(trial_label) != len(data_path):
            raise ValueError('Trial label and data path is not matched. Confirm your data.')

        # Import raw data
        self.raw_data = bf.read_data(trial_label=trial_label, data_path=data_path,
                                     fps=self.fps, time_bin=self.time_bin)

        # Match data size
        self.data = bf.match_data_size(data=self.raw_data)

        # Extract features
        self.data_features = bf.extract_features(data=self.data, q=self.q)

        # Reduce dimensionality
        self.data_features = bf.reduce_dimension(dataset=self.data_features, reducer=self.reducer,
                                                 fit_transform=True)

        # Calculate cluster metrics
        self.data_cluster_centroid = bf.calc_centroid(dataset=self.data_features)
        self.data_cluster_radius = bf.calc_radius(dataset=self.data_features)
        self.data_polygon_area = bf.calculate_cluster_polygon_area(centroid=self.data_cluster_centroid)

        # Calculate cluster centroid distance
        self.data_cluster_centroid_distance = (
            bf.calculate_cluster_centroid_distance(centroid=self.data_cluster_centroid))

        # Save cluster metrics
        df_data_cluster_centroid = pd.DataFrame.from_dict(self.data_cluster_centroid, orient='index')
        df_data_cluster_radius = pd.DataFrame.from_dict(self.data_cluster_radius, orient='index')
        df_data_polygon_area = pd.DataFrame.from_dict({'polygon_area': self.data_polygon_area}, orient='index')
        df_data_cluster_centroid_distance = pd.DataFrame.from_dict(self.data_cluster_centroid_distance, orient='index')

        df_data_cluster_centroid.to_csv(path_or_buf=os.path.join(epoch_storage, 'Cluster_centroid_coord.csv'))
        df_data_cluster_radius.to_csv(path_or_buf=os.path.join(epoch_storage, 'Cluster_radius.csv'))
        df_data_polygon_area.to_csv(path_or_buf=os.path.join(epoch_storage, 'Polygon_area.csv'))
        df_data_cluster_centroid_distance.to_csv(path_or_buf=os.path.join(epoch_storage, 'Cluster_distance.csv'))

    def run_visualize(self, trial_label: list, data_path: str):
        # Run visualization

        # Make random postfix
        timestamp = str(int(time.time()))  # Random postfix for experiment storage

        # Reset data
        self.__init_data__()

        # Reset reducer
        self.__init_reducer__()

        # Make experiment epoch storage
        epoch_storage = os.path.join(self.storage_dir, 'Experiment_Visualize_' + timestamp)
        if not os.path.exists(epoch_storage):
            os.makedirs(epoch_storage)

        # Prepare dataset
        self.prepare_dataset(trial_label=trial_label, data_path=data_path, epoch_storage=epoch_storage)

        # Visualize
        self.visualize(epoch_storage)

    def visualize(self, epoch_storage: str):
        # Check color theme
        if len(self.color_theme) < len(self.data_features):
            # Fill blank colors
            for _ in range(len(self.data_features) - len(self.color_theme)):
                self.color_theme.append("#%06x" % random.randint(0, 0xFFFFFF))

        # Data
        fig = go.Figure()  # Figure

        for idx, key in enumerate(self.data_features.keys()):
            # Legend
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name=str(key),
                                       marker=dict(size=10, color=self.color_theme[idx], symbol='circle')))

            # Cluster radius
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                       name=str(key) + ' radius = ' + str(self.data_cluster_radius[key]),
                                       marker=dict(size=10, color=self.color_theme[idx], symbol='circle')))

            # Display data
            temp_df_data_features = pd.DataFrame(self.data_features[key], columns=['dim1', 'dim2', 'dim3'])
            fig.add_trace(go.Scatter3d(x=temp_df_data_features['dim1'],
                                       y=temp_df_data_features['dim2'],
                                       z=temp_df_data_features['dim3'],
                                       mode='markers',
                                       marker=dict(
                                           symbol='circle',
                                           size=3,
                                           color=self.color_theme[idx],
                                           # choose a colorscale
                                           opacity=0.6),
                                       name='Cluster' + str(key),
                                       showlegend=False))

        data_centroid_labels, data_centroid_coords = group_embedding(self.data_cluster_centroid,
                                                                     close_scatter_line=True)
        df_emb_centroid = pd.DataFrame(
            np.concatenate((data_centroid_coords, np.array(data_centroid_labels).reshape(-1, 1)), axis=-1),
            columns=['dim1', 'dim2', 'dim3', 'label'])

        # Display centroid
        fig.add_trace(go.Scatter3d(x=df_emb_centroid['dim1'],
                                   y=df_emb_centroid['dim2'],
                                   z=df_emb_centroid['dim3'],
                                   mode='lines+markers',
                                   marker=dict(
                                       symbol='circle-open',
                                       size=10,
                                       color=df_emb_centroid['label'],
                                       colorscale=self.color_theme[:len(np.unique(df_emb_centroid['label']))],
                                       opacity=1),
                                   line=dict(
                                       width=5, color=self.line_color),
                                   name='Cluster centroid',
                                   showlegend=False))

        # Display centroid polygon hinge
        if self.show_hinge:
            draw_polygon_hinge(fig=fig, centroid=self.data_cluster_centroid, style=dict(line_color=self.line_color))

        # Display centroid labels
        if self.floating_labels:
            # Check label validity
            if (len(df_emb_centroid) - 1) != len(self.floating_labels):
                # Compare with len(df_emb_centroid)-1, because DataFrame for centroid coordinates
                # contains first cluster centroid in twice to close the representational polygon.
                raise ValueError('Failed matching floating labels to the clusters. Confirm your labels.')

            if len(self.floating_labels) != len(self.floating_labels_colors):
                raise ValueError('Failed matching floating labels to the floating label colors. Confirm your labels.')

            # Add labels
            annotations = []
            for i in range(len(df_emb_centroid) - 1):
                annotations.append(dict(showarrow=False,
                                        x=df_emb_centroid.iloc[i, 0],
                                        y=df_emb_centroid.iloc[i, 1],
                                        z=df_emb_centroid.iloc[i, 2],
                                        text=self.floating_labels[i],
                                        font=dict(color=self.floating_labels_colors[i],
                                                  size=self.floating_labels_fontsize)))
            fig.update_layout(scene=dict(annotations=annotations))

        for idx, (key, items) in enumerate(self.data_cluster_centroid_distance.items()):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                       name=str(key) + ': ' + str(items),
                                       marker=dict(size=10, color=self.line_color, symbol='circle')))

        fig.update_layout(title="Visualize cluster", font=dict(size=15),
                          scene_aspectmode='cube',
                          scene=dict(xaxis_title="Dim 1",
                                     yaxis_title="Dim 2",
                                     zaxis_title="Dim 3",
                                     xaxis=dict(showline=True, linewidth=5, linecolor='black', ticks='outside',
                                                # tickwidth=5, dtick=10,
                                                gridcolor="#d1d1d1", showbackground=False),
                                     yaxis=dict(showline=True, linewidth=5, linecolor='black', ticks='outside',
                                                # tickwidth=5, dtick=10,
                                                gridcolor="#d1d1d1", showbackground=False),
                                     zaxis=dict(showline=True, linewidth=5, linecolor='black', ticks='outside',
                                                # tickwidth=5, dtick=10,
                                                gridcolor="#d1d1d1", showbackground=False)),
                          autosize=True)

        if self.rotating:
            # Camera eye
            x_eye = -2.75
            y_eye = 2.5
            z_eye = 1

            # Add frames
            frames = []
            for t in np.arange(0, 6.26, 0.1):
                xe, ye, ze = rotate_cam_z(x_eye, y_eye, z_eye, -t)
                frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

            fig.update_layout(legend=dict(yanchor="top", y=0.99,
                                          xanchor="left", x=0.01),
                              legend_title_text='Color codes (Visualization)')

            fig.update_layout(scene=dict(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye))),
                              updatemenus=[dict(type='buttons', showactive=False, xanchor='right', yanchor='top',
                                                buttons=[dict(label='Play',
                                                              method='animate',
                                                              args=[None, dict(frame=dict(duration=50,
                                                                                          redraw=True),
                                                                               transition=dict(duration=0,
                                                                                               easing='linear'),
                                                                               fromcurrent=True,
                                                                               mode='animate')])])])

            # Add animations
            fig.frames = frames

        # Show figure
        fig.show()

        # Save results
        fig_name = 'Fig_Visualize.html'
        fig.write_html(os.path.join(epoch_storage, fig_name))


def group_embedding(centroids: dict, close_scatter_line: bool = False):
    # Returns
    labels, coordinates = [], []

    # Grouping embeddings
    for l, coord in centroids.items():
        labels.append(ord(l))
        coordinates.append(coord)

    # Add closing coords
    if close_scatter_line:
        labels.append(labels[0])
        coordinates.append(coordinates[0])

    return labels, coordinates


def rotate_cam_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def to_rgba(hex_color: list, alpha: float = 1.0):
    # Remove the hash at the beginning if it's there
    hex_color = hex_color.lstrip('#')

    # Convert the hex string to an integer
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return rgb + (alpha,)


def draw_polygon_hinge(fig: go.Figure, centroid: dict, style: dict):
    # Check validity
    if len(centroid) < 3:
        warnings.warn('Cannot make hinge. (This message can be ignored.)')
        return

    # Get centroid values
    values = list(centroid.values())

    # Set v0
    v0 = values.pop(0)

    # Remove first and last values
    values = values[1:-1]

    # Draw hinge
    for i in range(len(values)):
        fig.add_trace(go.Scatter3d(x=[v0[0], values[i][0]],
                                   y=[v0[1], values[i][1]],
                                   z=[v0[2], values[i][2]],
                                   mode='lines',
                                   line=dict(
                                       width=5, color=style['line_color'], dash='dash'),
                                   showlegend=False))
