def check_batch_dimension(self, model, dataloaders, test_val=2):
    r"""
    Verifies that the provided model loads the data correctly. We do this by setting the
    loss to be something trivial (e.g. the sum of all outputs of example i), running the
    backward pass all the way to the input, and ensuring that we only get a non-zero gradient
    on the i-th input.
    See details at http://karpathy.github.io/2019/04/25/recipe/.
    """
    model.zero_grad()
    model.eval()

    # copy properties for forward overrides
    self.copy_trainer_model_properties(model)

    # disable gradients to save memory
    torch.set_grad_enabled(False)

    # bookkeeping
    outputs = []

    # run validation
    for dataloader_idx, dataloader in enumerate(dataloaders):
        dl_outputs = []

        # on TPU we have to wrap it under the ParallelLoader
        if self.use_tpu:
            device = xm.xla_device()
            dataloader = xla_pl.ParallelLoader(dataloader, [device])
            dataloader = dataloader.per_device_loader(device)

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            # -----------------
            # RUN BATCH CHECK
            # -----------------
            data = batch[0]
            data.requires_grad_()
            output = model(data)
            loss = output[test_val].sum()
            loss.backward()

            assert loss != 0, "Loss should be greater than zero."
            assert (data.grad[test_val] != 0).any(), "The test input gradient should be zero."
            assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all(), \
                "There are nonzero gradients in the batch, when they should all be zero."

        outputs.append(dl_outputs)

    eval_results = {}

    # with a single dataloader don't pass an array
    if len(dataloaders) == 1:
        outputs = outputs[0]

    # give model a chance to do something with the outputs (and method defined)
    if isinstance(model, (LightningDistributedDataParallel, LightningDataParallel)):
        model = model.module

    # enable train mode again
    model.train()

    # enable gradients to save memory
    torch.set_grad_enabled(True)

    return eval_results

    # python -m coverage run --source pytorch_lightning -m py.test pytorch_lightning tests pl_examples -v --doctest-modules --flake8 -k test_simple_cpu -s