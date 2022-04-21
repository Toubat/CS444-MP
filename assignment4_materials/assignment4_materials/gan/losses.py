import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss, mse_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    loss_real = bce_loss(logits_real, torch.ones_like(logits_real), reduction='mean')
    loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake), reduction='mean')
    loss = loss_real + loss_fake

    ##########       END      ##########

    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    loss = bce_loss(logits_fake, torch.ones_like(logits_fake), reduction='mean')

    ##########       END      ##########

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    loss_real = 0.5 * torch.mean((scores_real - 1) ** 2)
    loss_fake = 0.5 * torch.mean(scores_fake ** 2)
    loss = loss_real + loss_fake

    ##########       END      ##########

    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    loss = 0.5 * torch.mean((scores_fake - 1) ** 2)

    ##########       END      ##########

    return loss


def w_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Wasserstein GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    loss = -torch.mean(scores_real) + torch.mean(scores_fake)

    ##########       END      ##########

    return loss

def w_generator_loss(scores_fake):
    """
    Computes the Wasserstein GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    loss = -torch.mean(scores_fake)

    ##########       END      ##########

    return loss