import brain.decoder.ccgp as ccgp

# Paths
trial_label = [
    'A',
    'B',
    'C',
    'D',
]
data_path = [
    # TEST TRIAL
    'path 1',
    'path 2',
    'path 3',
    'path 4'

]
storage_dir = 'storage directory'

# Color table
# Littermate: #19cceb (Train)
# Littermate2: #0c788a (Train)
# Littermate (Test): #08b6d4

# Familiar: #fc9f9f (Train)
# Familiar2: #f94545 (Train)
# Familiar (Test): #ed7777

# Novel: #8faadc (Train)
# Novel2: #4975c7 (Train)
# Novel (Test): #6681b3

# Make CCGP object
ccgp_object = ccgp.CCGP()

# Set parameters
ccgp_object.storage_dir = storage_dir
ccgp_object.null_mode = False
ccgp_object.percent_mode = True
ccgp_object.EPOCHS = 1
ccgp_object.q = 5
ccgp_object.fps = 20
ccgp_object.time_bin = 200  # ms
ccgp_object.dim_reduction_method = 'pca'
ccgp_object.train_org = [['A'], ['B']]  # train [[Labels 0][Label 1]]
ccgp_object.test_org = [['C'], ['D']]  # test [[Label 0][Label 1]]

# Visualization color coding
# ccgp_object.color_train_label_0 = ''  # Train
# ccgp_object.color_train_label_1 = ''
# ccgp_object.color_line_train = '' # Train line
#
# ccgp_object.color_test_label_0 = ''  # Test
# ccgp_object.color_test_label_1 = ''
# ccgp_object.color_line_test = '' # Test line

# Run CCGP
ccgp_object.run_ccgp(trial_label, data_path)
