import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError
    
  

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############
        distribution = self.action_distribution(observations)
        sampled_actions = distribution.sample()

        if return_log_prob:
            log_probs = distribution.log_prob(sampled_actions)
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        else:
            log_probs = None
        
        
        # with torch.no_grad():
        #     sampled_actions = self.action_distribution(observations).sample().cpu().numpy()

        #######################################################
        #########          END YOUR CODE.          ############
        # if return_log_prob:
        #     return sampled_actions, log_probs
        # return sampled_actions
        return sampled_actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy() if return_log_prob else None




class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network
        #self.is_disc_action = is_disc_action  # Assign is_disc_action


    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        distribution = ptd.Categorical(logits=self.network(observations))
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.
        """
        nn.Module.__init__(self)
        self.network = network
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        self.log_std = nn.Parameter(data=torch.zeros(action_dim))
        #######################################################
        #########          END YOUR CODE.          ############

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]

        The return value contains the standard deviations for each dimension
        of the policy's actions. It can be computed from self.log_std
        """
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        std = torch.exp(self.log_std)
        #######################################################
        #########          END YOUR CODE.          ############
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """
        #######################################################
        #########   YOUR CODE HERE - 2-4 lines.    ############
        #loc = self.network(observations)
        #distribution = torch.distributions.MultivariateNormal(loc, covariance_matrix=torch.eye(loc.shape[0], loc.shape[0]))
        loc = self.network(observations)
        
        if observations.shape[0] == 1:
            # For the pendulum environment, handle the special case of batch size 1
            scale_tril = torch.diag_embed(torch.exp(self.log_std)).unsqueeze(0)
        else:
            scale_tril = torch.diag_embed(torch.exp(self.log_std))

        distribution = torch.distributions.MultivariateNormal(loc, scale_tril=scale_tril)
        
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution
