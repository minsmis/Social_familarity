import os

import time
import random
import numpy as np
import joblib
import mat73 as mat
import plotly.graph_objects as go
import pandas as pd

import umap
import sklearn
from sklearn import svm  # import sklearn svm package
from sklearn.preprocessing import StandardScaler

import brain.feature.social as bs


class CCGP:
    def __init__(self):
        super(CCGP, self).__init__()

        # Storage path
        self.storage_dir = ''

        # Init
        self.__init_parameters__()
        self.__init_data__()
        self.__init_metrics__()
        self.__init_model__()
        self.__init_reducer__()
        self.__init_visualization__()

        # Experiment performance
        self.accuracy_list = []

    def __init_parameters__(self):
        # Preprocessing parameters
        self.null_mode = False  # True: Test Null decoder, False: Test genuine decoder
        self.percent_mode = True  # Report accuracy in percent?
        self.EPOCHS = 5  # Number of epochs to repeat decoder performance test
        self.q = 5
        self.fps = 20
        self.time_bin = 100  # ms
        self.train_org = []
        self.test_org = []
        self.train_ratio = 0.75
        self.dim_reduction_method = 'pca'
        self.n_dim = 3

    def __init_data__(self):
        # Data
        self.raw_data = {}

        self.data = {}

        self.feature_dataset = {}

        self.train_class = {}
        self.test_class = {}

        self.train_dataset = []
        self.train_label = []
        self.test_dataset = []
        self.test_label = []

    def __init_metrics__(self):
        self.cluster_centroid = {}
        self.cluster_radius = {}
        self.cluster_centroid_distance = {}

    def __init_model__(self):
        # Model
        self.model = svm.SVC(kernel='linear', C=0.5, verbose=True)

    def __init_reducer__(self):
        # Reducer
        self.reducer = bs.make_reducer(method=self.dim_reduction_method, dim=self.n_dim)

    def __init_visualization__(self):
        # Visualization color
        self.color_train_label_0 = '#8faadc'
        self.color_train_label_1 = '#fc9f9f'
        self.color_test_label_0 = '#6681b3'
        self.color_test_label_1 = '#c65c54'

        self.color_line_train = '#004812'
        self.color_line_test = '#FF5520'

    def prepare_dataset(self, trial_label: list, data_path: str, epoch_storage: str):
        # Prepare dataset for CCGP

        # Check validity: Label-Data match
        if len(trial_label) != len(data_path):
            raise ValueError('Trial label and data path is not matched. Confirm your data.')

        # Check validity: Trial length
        if len(trial_label) != 4 and len(data_path) != 4:
            raise ValueError('This decoder is compatible only for 4 trial conditions. Confirm your data.')

        # Check validity: Class organization
        merged_org = list(np.concatenate([self.train_org, self.test_org], axis=0))
        if len(np.array(merged_org).reshape(-1)) != 4:
            raise ValueError(
                'This decoder is compatible only for binary classes. Confirm your class organizations.')

        # Import raw data
        self.raw_data = bs.read_data(trial_label=trial_label, data_path=data_path,
                                     fps=self.fps, time_bin=self.time_bin)

        # Select data will use
        self.raw_data = bs.select_data(data=self.raw_data, org=merged_org)

        # Match dataset size
        self.data = bs.match_data_size(data=self.raw_data)

        # Extract features
        self.feature_dataset = bs.extract_features(data=self.data, q=self.q)

        # Reduce dimensionality
        self.feature_dataset = bs.reduce_dimension(dataset=self.feature_dataset, reducer=self.reducer,
                                                   fit_transform=True)

        # Calculate cluster metrics
        self.cluster_centroid = bs.calc_centroid(self.feature_dataset)
        self.cluster_radius = bs.calc_radius(self.feature_dataset)

        # Calculate cluster centroid distance for org pairs
        train_distance_label, train_distance = bs.calculate_org_centroid_distance(centroid=self.cluster_centroid,
                                                                                  org=self.train_org)
        test_distance_label, test_distance = bs.calculate_org_centroid_distance(centroid=self.cluster_centroid,
                                                                                org=self.test_org)
        self.cluster_centroid_distance[train_distance_label] = train_distance
        self.cluster_centroid_distance[test_distance_label] = test_distance

        # Save cluster metrics
        df_cluster_centroid = pd.DataFrame.from_dict(self.cluster_centroid, orient='index')
        df_cluster_radius = pd.DataFrame.from_dict(self.cluster_radius, orient='index')
        df_cluster_centroid_distance = pd.DataFrame.from_dict(self.cluster_centroid_distance, orient='index')

        df_cluster_centroid.to_csv(path_or_buf=os.path.join(epoch_storage, 'Cluster_centroid_coord.csv'))
        df_cluster_radius.to_csv(path_or_buf=os.path.join(epoch_storage, 'Cluster_radius.csv'))
        df_cluster_centroid_distance.to_csv(path_or_buf=os.path.join(epoch_storage, 'Cluster_distance.csv'))

        # Make data class
        self.train_class, _ = bs.make_class(data=self.feature_dataset, class_org=self.train_org)
        self.test_class, _ = bs.make_class(data=self.feature_dataset, class_org=self.test_org)

        # Make dataset
        self.train_dataset, self.train_label = bs.make_dataset(data_class=self.train_class, null_mode=self.null_mode)
        self.test_dataset, self.test_label = bs.make_dataset(data_class=self.test_class)

    def train(self):
        # Train decoder with embedded dataset
        self.model = self.model.fit(self.train_dataset, self.train_label)

    def test(self):
        # Variables
        accuracy = 0
        correct = 0

        # Predict test_data
        prediction = self.model.predict(self.test_dataset)

        # Compare prediction with test_label
        for predict, label in zip(prediction, self.test_label):
            if predict == label:
                correct += 1

        # Calculate accuracy
        total = len(prediction)
        if self.percent_mode is True:
            accuracy = 100 * (correct / total)
        if self.percent_mode is False:
            accuracy = correct / total

        # Report accuracy
        report_accuracy(accuracy, self.percent_mode)

        return accuracy

    def run_ccgp(self, trial_label: list, data_path: str):
        # Run CCGP experiment

        # Make random postfix
        timestamp = str(int(time.time()))  # Random postfix for experiment storage

        for epoch in range(self.EPOCHS):
            # Notice current epoch
            notice_current_epoch(epoch + 1)

            # Reset data
            self.__init_data__()

            # Reset decoder
            self.__init_model__()

            # Reset reducer
            self.__init_reducer__()

            # Make experiment epoch storage
            epoch_storage = os.path.join(self.storage_dir, 'Experiment_CCGP_' + timestamp, 'EPOCH_' + str(epoch + 1))
            if not os.path.exists(epoch_storage):
                os.makedirs(epoch_storage)

            # Prepare dataset
            self.prepare_dataset(trial_label=trial_label, data_path=data_path, epoch_storage=epoch_storage)

            # Train decoder
            self.train()

            # Save decoder
            self.save_model(epoch_storage)

            # Test decoder
            accuracy = self.test()

            # Store accuracy for experiment
            self.accuracy_list.append(accuracy)

            # Visualize
            self.visualize(epoch, accuracy, epoch_storage)

        # Report experiment result: CCGP performance
        self.report()

    def visualize(self, epoch: int, accuracy: float, epoch_storage: str):
        # Hyperplane
        min_x, max_x = np.min(self.test_dataset[:, 0]), np.max(self.test_dataset[:, 0])
        min_y, max_y = np.min(self.test_dataset[:, 1]), np.max(self.test_dataset[:, 1])
        mean_z = np.mean(self.test_dataset[:, 2])
        tmp_x = np.linspace(round(abs(min_x) * -1.5), round(abs(max_x) * 1.5), 30)
        tmp_y = np.linspace(round(abs(min_y) * -1.5), round(abs(max_y) * 1.5), 30)
        x, y = np.meshgrid(tmp_x, tmp_y)

        w = self.model.coef_[0]
        i = self.model.intercept_[0]
        d = -i / w[-1]
        a = -w[0] / w[-1]
        b = -w[1] / w[-1]
        z = d + (a * x) + (b * y)

        # Display
        df_emb_train = pd.DataFrame(self.train_dataset, columns=['dim1', 'dim2', 'dim3'])
        df_emb_train['label'] = self.train_label

        train_centroid_labels, train_centroid_coords = group_embedding(self.cluster_centroid, self.train_org)
        df_emb_centroid_train = pd.DataFrame(
            np.concatenate((train_centroid_coords, np.array(train_centroid_labels).reshape(-1, 1)), axis=-1),
            columns=['dim1', 'dim2', 'dim3', 'label'])

        df_emb_test = pd.DataFrame(self.test_dataset, columns=['dim1', 'dim2', 'dim3'])
        df_emb_test['label'] = self.test_label

        test_centroid_labels, test_centroid_coords = group_embedding(self.cluster_centroid, self.test_org)
        df_emb_centroid_test = pd.DataFrame(
            np.concatenate((test_centroid_coords, np.array(test_centroid_labels).reshape(-1, 1)), axis=-1),
            columns=['dim1', 'dim2', 'dim3', 'label'])

        # Legend
        label_train_0 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name='Train::0',
                                     marker=dict(size=10, color=self.color_train_label_0, symbol='circle-open'))
        label_train_1 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name='Train::1',
                                     marker=dict(size=10, color=self.color_train_label_1, symbol='circle-open'))
        label_test_0 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name='Test::0',
                                    marker=dict(size=10, color=self.color_test_label_0, symbol='circle'))
        label_test_1 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name='Test::1',
                                    marker=dict(size=10, color=self.color_test_label_1, symbol='circle'))

        # Cluster radius
        train_org = np.array(self.train_org).reshape(-1)
        test_org = np.array(self.test_org).reshape(-1)
        label_radius_train_0 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                            name='Train::0 radius = ' + str(self.cluster_radius[train_org[0]]),
                                            marker=dict(size=10, color=self.color_train_label_0, symbol='circle-open'))
        label_radius_train_1 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                            name='Train::1 radius = ' + str(self.cluster_radius[train_org[1]]),
                                            marker=dict(size=10, color=self.color_train_label_1, symbol='circle-open'))
        label_radius_test_0 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                           name='Test::0 radius = ' + str(self.cluster_radius[test_org[0]]),
                                           marker=dict(size=10, color=self.color_test_label_0, symbol='circle'))
        label_radius_test_1 = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                           name='Test::1 radius = ' + str(self.cluster_radius[test_org[1]]),
                                           marker=dict(size=10, color=self.color_test_label_1, symbol='circle'))

        # Cluster centroid distance
        label_distance_train = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                            name='Train::' + str(train_org[0]) + str(train_org[1]) + ': ' + str(
                                                self.cluster_centroid_distance[str(train_org[0]) + str(train_org[1])]),
                                            marker=dict(size=10, color=self.color_line_train,
                                                        symbol='circle-open'))
        label_distance_test = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                           name='Test::' + str(test_org[0]) + str(test_org[1]) + ': ' + str(
                                               self.cluster_centroid_distance[str(test_org[0]) + str(test_org[1])]),
                                           marker=dict(size=10, color=self.color_line_test,
                                                       symbol='circle'))

        # Data
        scatter_emb_train = go.Scatter3d(x=df_emb_train['dim1'], y=df_emb_train['dim2'], z=df_emb_train['dim3'],
                                         mode='markers',
                                         marker=dict(
                                             symbol='circle-open',
                                             size=3,
                                             color=df_emb_train['label'],
                                             # set color to an array/list of desired values
                                             colorscale=[self.color_train_label_0, self.color_train_label_1],
                                             # choose a colorscale [Label 0, Label 1]
                                             opacity=0.6),
                                         name='Train',
                                         showlegend=False)

        scatter_emb_centroid_train = go.Scatter3d(x=df_emb_centroid_train['dim1'],
                                                  y=df_emb_centroid_train['dim2'],
                                                  z=df_emb_centroid_train['dim3'],
                                                  mode='lines+markers',
                                                  marker=dict(
                                                      symbol='circle-open',
                                                      size=10,
                                                      color=df_emb_centroid_train['label'],
                                                      colorscale=[self.color_train_label_0,
                                                                  self.color_train_label_1],
                                                      opacity=1),
                                                  line=dict(
                                                      width=5, color=self.color_line_train),
                                                  name='Train centroid',
                                                  showlegend=False)

        scatter_emb_test = go.Scatter3d(x=df_emb_test['dim1'], y=df_emb_test['dim2'], z=df_emb_test['dim3'],
                                        mode='markers',
                                        marker=dict(
                                            symbol='circle',
                                            size=3,
                                            color=df_emb_test['label'],  # set color to an array/list of desired values
                                            colorscale=[self.color_test_label_0, self.color_test_label_1],
                                            # choose a colorscale [Label 0, Label 1]
                                            opacity=0.6),
                                        name='Test',
                                        showlegend=False)

        scatter_emb_centroid_test = go.Scatter3d(x=df_emb_centroid_test['dim1'],
                                                 y=df_emb_centroid_test['dim2'],
                                                 z=df_emb_centroid_test['dim3'],
                                                 mode='lines+markers',
                                                 marker=dict(
                                                     symbol='circle',
                                                     size=10,
                                                     color=df_emb_centroid_test['label'],
                                                     colorscale=[self.color_test_label_0,
                                                                 self.color_test_label_1],
                                                     opacity=1),
                                                 line=dict(
                                                     width=5, color=self.color_line_test),
                                                 name='Test centroid',
                                                 showlegend=False)

        hyperplane = go.Surface(z=z, x=x, y=y,
                                colorscale=['#696969' for i in range(len(z))],
                                opacity=0.5,
                                name="Hyperplane")

        fig = go.Figure()
        fig.add_trace(scatter_emb_train)
        fig.add_trace(scatter_emb_centroid_train)
        fig.add_trace(scatter_emb_test)
        fig.add_trace(scatter_emb_centroid_test)

        fig.add_trace(label_train_0)  # Custom legends
        fig.add_trace(label_train_1)

        fig.add_trace(label_test_0)
        fig.add_trace(label_test_1)

        fig.add_trace(label_radius_train_0)
        fig.add_trace(label_radius_train_1)
        fig.add_trace(label_radius_test_0)
        fig.add_trace(label_radius_test_1)

        fig.add_trace(label_distance_train)
        fig.add_trace(label_distance_test)

        fig.add_trace(hyperplane)
        fig.update_layout(title="CCGP = " + str(accuracy), font=dict(size=15),
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
                                                range=[mean_z - 3, mean_z + 3],
                                                # tickwidth=5, dtick=10,
                                                gridcolor="#d1d1d1", showbackground=False)),
                          autosize=True)
        fig.update_layout(legend=dict(yanchor="top", y=0.99,
                                      xanchor="left", x=0.01),
                          legend_title_text='Color codes (CCGP)')

        fig_summary = go.Figure()
        fig_summary.add_trace(scatter_emb_centroid_train)
        fig_summary.add_trace(scatter_emb_centroid_test)

        fig_summary.add_trace(label_train_0)  # Custom legends
        fig_summary.add_trace(label_train_1)

        fig_summary.add_trace(label_test_0)
        fig_summary.add_trace(label_test_1)

        fig_summary.add_trace(label_radius_train_0)
        fig_summary.add_trace(label_radius_train_1)
        fig_summary.add_trace(label_radius_test_0)
        fig_summary.add_trace(label_radius_test_1)

        fig_summary.add_trace(label_distance_train)
        fig_summary.add_trace(label_distance_test)

        fig_summary.add_trace(hyperplane)

        fig_summary.update_layout(title="CCGP = " + str(accuracy), font=dict(size=15),
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
                                                        range=[mean_z - 3, mean_z + 3],
                                                        # tickwidth=5, dtick=10,
                                                        gridcolor="#d1d1d1", showbackground=False)),
                                  autosize=True)
        fig_summary.update_layout(legend=dict(yanchor="top", y=0.99,
                                              xanchor="left", x=0.01),
                                  legend_title_text='Color codes (CCGP)')

        fig.show()
        fig_summary.show()

        # Save results
        fig_name = 'Fig_EPOCH_' + str(epoch + 1) + '.html'
        fig.write_html(os.path.join(epoch_storage, fig_name))
        fig_summary_name = 'Fig_Summary_EPOCH_' + str(epoch + 1) + '.html'
        fig.write_html(os.path.join(epoch_storage, fig_summary_name))

    def save_model(self, storage_dir):
        # Make storage path if not exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        joblib.dump(self.model, os.path.join(storage_dir, 'trained_model.pkl'))

    def report(self):
        # This function reports averaged performances for all EXPERIMENT_EPOCHS
        performance = np.average(self.accuracy_list)
        if self.percent_mode is True:
            template = "\n-----\nPERFORMANCE: {}%\n-----\n"
            print(template.format(performance))
        if self.percent_mode is False:
            template = "\n-----\nPERFORMANCE: {}\n-----\n"
            print(template.format(performance))


def group_embedding(centroids: dict, org: list):
    # Returns
    labels, coordinates = [], []

    # Get org
    org_reshaped = np.array(org).reshape(-1)

    # Grouping embeddings
    for l, coord in centroids.items():
        if l in org_reshaped:
            labels.append(ord(l))
            coordinates.append(coord)

    # Add closing coords
    # labels.append(labels[0])
    # coordinates.append(coordinates[0])

    return labels, coordinates


def notice_current_epoch(epoch: int):
    template = "----- Epoch: {} -----"
    print(template.format(epoch))


def report_accuracy(accuracy: float, percent_mode: bool = False):
    if percent_mode is True:
        template = "\n-----\nACCURACY: {}%\n-----\n"
        print(template.format(accuracy))
    if percent_mode is False:
        template = "\n-----\nACCURACY: {}\n-----\n"
        print(template.format(accuracy))
