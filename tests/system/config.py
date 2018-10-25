from sharp.config.spec import SharpConfigBase


class SharpConfig(SharpConfigBase):

    # [Data]
    output_dir = "data/processed"
    raw_data_dir = "data/raw"
    reference_channel = "L2 - E13_extract"

    # [Main]
    config_id = "test"
    num_thresholds = 6

    # [NeuralNet]
    num_layers = 1
    num_units_per_layer = 8
    num_epochs = 2

    def get_tasks(self):
        from sharp.tasks.neuralnet.apply import ApplyRNN
        from sharp.tasks.plot.misc.training import PlotValidLoss

        return super().get_tasks() + (ApplyRNN(), PlotValidLoss())
