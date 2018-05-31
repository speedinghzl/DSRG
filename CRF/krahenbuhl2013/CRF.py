from krahenbuhl2013.wrapper import DenseCRF


def CRF(image, unary, maxiter=10, scale_factor=1.0, color_factor=13):
    """
        Performs infernce in a fully connected CRF with gaussian potentials

        Inputs:
            1. image: an RGB image represented as a numpy matrix of size H x W x 3,
                      where H is the image height and W is the image width.
                      The values should be normalized to be within range of [0..256].
            2. prob: predicted probabilities represented as numpy matrix of size H x W x M,
                     where M is a number of classes.
            3. Number of iterations for the inference algorithm (default: 10).

        Outputs the most likely configuration of pixel labels.
    """

    assert(image.shape[:2] == unary.shape[:2])

    H, W = image.shape[:2]
    nlables = unary.shape[2]

    # initialize CRF
    crf = DenseCRF(W, H, nlables)

    # set unary potentials
    crf.set_unary_energy(-unary.ravel().astype('float32'))

    # set pairwise potentials
    crf.add_pairwise_energy(10, 80 / scale_factor, 80 / scale_factor, color_factor, color_factor, color_factor,
                            3, 3 / scale_factor, 3 / scale_factor, image.ravel().astype('ubyte'))

    # run inference
    prediction = crf.inference(maxiter).reshape((H, W, nlables))

    return prediction
