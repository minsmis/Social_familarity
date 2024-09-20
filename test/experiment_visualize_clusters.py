import brain.visualize.clusters as bv

# Paths
trial_label = [
    'A',
    'B',
    'C',
    'D'
]
data_path = [
    # TEST TRIAL
    'path1',
    'path2',
    'path3',
    'path4'
]

storage_dir = 'storage directory'

# Make XOR object
vis_object = bv.Clusters()

# Set parameters
vis_object.storage_dir = storage_dir
vis_object.q = 5
vis_object.fps = 20
vis_object.time_bin = 200  # ms
vis_object.dim_reduction_method = 'pca'
vis_object.rotating = False
vis_object.show_hinge = True

# Visualization color coding
# vis_object.color_theme = []  # ['Hex color for label A', 'Hex color for label B', ...]
# vis_object.line_color = '#.....' # Hex color for connecting line

# Visualization floating labels
vis_object.floating_labels = ['<b>Label1</b>', '<b>Label2</b>', '<b>Label3</b>',
                              '<b>Label4</b>']  # Bold = Add <b>...</b> tags
vis_object.floating_labels_fontsize = 36
vis_object.floating_labels_colors = ['lightsteelblue', 'mediumslateblue', 'palegreen', 'lavender']
# ['Hex color for label A', 'Hex color for label B', ...]

# Run XOR
vis_object.run_visualize(trial_label, data_path)
