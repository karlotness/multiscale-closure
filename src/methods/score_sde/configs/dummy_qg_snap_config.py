# Based on config in default_cifar10_configs.py and cifar10_ddpmpp.py
import ml_collections

def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128 # Reduce batch size? (/4)
    training.n_iters = 1300001 # NUM_BATCHES
    training.snapshot_freq = 50000 # Set to 1
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.n_jitted_steps = 5 # Batches per epoch
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'qg_snap'
    data.image_size = 64
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 2

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.1
    model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42

    return config

def build_dummy_qg_snap_config(batch_size, num_epochs, batches_per_epoch, lr):
    config = get_default_configs()
    # Override basic values
    config.training.batch_size = batch_size
    config.training.n_iters = num_epochs
    config.training.snapshot_freq = 1
    config.training.n_jitted_steps = batches_per_epoch
    config.optim.lr = lr

    # training
    training = config.training
    training.sde = 'vpsde'
    training.continuous = False
    training.reduce_mean = True

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'ancestral_sampling'
    sampling.corrector = 'none'

    # data
    data = config.data
    data.centered = True

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'none'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.embedding_type = 'positional'
    model.fourier_scale = 16
    model.conv_size = 3

    return config
