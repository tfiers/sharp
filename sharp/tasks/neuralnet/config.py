from luigi import Config, IntParameter, FloatParameter


class NeuralNet(Config):
    # RNN architecture
    # ----------------
    num_layers: int = IntParameter(2)
    num_units_per_layer: int = IntParameter(20)

    # Training settings
    # -----------------
    reference_seg_extension: float = FloatParameter(0)
    # Reference segments are expanded at their leading edge, by the given
    # fraction of total segment duration (= approximate SWR duration). This
    # should encourage SWR 'prediction' in the optimisation procedure.

    chunk_duration: float = FloatParameter(0.3)
    # Length of a chunk, in seconds. Network weights are updated after each
    # chunk of training data has been processed.

    p_dropout: float = FloatParameter(0.4)
    # Probability that a random hidden unit's activation is set to 0 during a
    # training step. Should improve generalisation performance. Only relevant
    # for num_layers > 1.

    num_epochs: int = IntParameter()
    # How many times to pass over the training data when training an RNN.

    valid_fraction: float = FloatParameter(0.22)
    # How much of the training data to use for validation (estimation of
    # generalisation performance -- to choose net of epoch where this was
    # best). The rest of the data is used for training proper.


neural_net_config = NeuralNet()
