from sharp.config.spec import SharpConfigBase


class SharpConfig(SharpConfigBase):

    # [Data]
    raw_data_dir = "data/raw"
    output_dir = "data/processed"
    reference_channel = "L2 - E13_extract"
    toppyr_channel_ix = 2
    sr_channel_ix = 0

    # [Main]
    config_id = "test"
    num_thresholds = 6
    channel_combinations = {
        "all": (0, 1, 2),
        "pyr": (0,),
        "sr": (2,),
        "pyr+sr": (0, 2),
    }
    time_ranges = [(0.68, 1.2)]

    # [NeuralNet]
    num_layers = 1
    num_units_per_layer = 8
    num_epochs = 2

    # def get_tasks(self):
    #     # from sharp.tasks.neuralnet.apply import ApplyRNN
    #     # from sharp.tasks.plot.misc.training import PlotValidLoss
    #     #
    #     # return super().get_tasks() + (ApplyRNN(), PlotValidLoss())
