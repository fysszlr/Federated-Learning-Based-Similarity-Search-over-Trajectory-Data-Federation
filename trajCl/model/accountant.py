import numpy as np
import abc

class MomentsAccountant(object):
    """Privacy accountant which keeps track of moments of privacy loss."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, total_examples, moment_orders=32):
        """Initialize a MomentsAccountant.
        Args:
          total_examples: total number of examples.
          moment_orders: the order of moments to keep.
        """
        assert total_examples > 0
        self._total_examples = total_examples
        self._moment_orders = (moment_orders
                               if isinstance(moment_orders, (list, tuple))
                               else range(1, moment_orders + 1))
        self._max_moment_order = max(self._moment_orders)
        assert self._max_moment_order < 100, "The moment order is too large."
        self._log_moments = [0.0 for _ in self._moment_orders]

    @abc.abstractmethod
    def _compute_log_moment(self, sigma, q, moment_order):
        """Compute high moment of privacy loss.
        Args:
          sigma: the noise sigma, in the multiples of the sensitivity.
          q: the sampling ratio.
          moment_order: the order of moment.
        Returns:
          log E[exp(moment_order * X)]
        """
        pass

    def accumulate_privacy_spending(self, unused_eps_delta, sigma, num_examples):
        """Accumulate privacy spending.
        Args:
          unused_eps_delta: EpsDelta pair which can be tensors. Unused
            in this accountant.
          sigma: the noise sigma, in the multiples of the sensitivity (that is,
            if the l2norm sensitivity is k, then the caller must have added
            Gaussian noise with stddev=k*sigma to the result of the query).
          num_examples: the number of examples involved.
        Returns:
          a numpy operation for updating the privacy spending.
        """
        q = num_examples * 1.0 / self._total_examples
        for i in range(len(self._log_moments)):
            moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
            self._log_moments[i] += moment
        print("_log_moments:",self._log_moments)

    def _compute_delta(self, log_moments, eps):
        """Compute delta for given log_moments and eps.
        Args:
          log_moments: the log moments of privacy loss, in the form of pairs
            of (moment_order, log_moment)
          eps: the target epsilon.
        Returns:
          delta
        """
        min_delta = 1.0
        for moment_order, log_moment in log_moments:
            if np.isinf(log_moment) or np.isnan(log_moment):
                print("The %d-th order is inf or Nan" % moment_order)
                continue
            if log_moment < moment_order * eps:
                min_delta = min(min_delta,
                                np.exp(log_moment - moment_order * eps))
        return min_delta

    def _compute_eps(self, log_moments, delta):
        min_eps = float("inf")
        for moment_order, log_moment in log_moments:
            if np.isinf(log_moment) or np.isnan(log_moment):
                print("The %d-th order is inf or Nan" % moment_order)
                continue
            min_eps = min(min_eps, (log_moment - np.log(delta)) / moment_order)
        return min_eps

    def get_privacy_spent(self, eps):
        """Compute privacy spending in (e, d)-DP form for a single or list of eps."""
        log_moments_with_order = zip(self._moment_orders, self._log_moments)
        return self._compute_delta(log_moments_with_order, eps)


class GaussianMomentsAccountant(MomentsAccountant):
    """MomentsAccountant which assumes Gaussian noise."""

    def __init__(self, total_examples, moment_orders=32):
        """Initialization.
        Args:
          total_examples: total number of examples.
          moment_orders: the order of moments to keep.
        """
        super(GaussianMomentsAccountant, self).__init__(total_examples, moment_orders)
        self._binomial_table = self._generate_binomial_table(self._max_moment_order)

    def _generate_binomial_table(self, max_moment_order):
        table = np.zeros((max_moment_order + 1, max_moment_order + 1))
        for n in range(max_moment_order + 1):
            table[n, 0] = 1
            for k in range(1, n + 1):
                table[n, k] = table[n - 1, k - 1] + table[n - 1, k]
        return table

    def _differential_moments(self, sigma, s, t):
        """Compute 0 to t-th differential moments for Gaussian variable."""
        assert t <= self._max_moment_order, ("The order of %d is out "
                                             "of the upper bound %d."
                                             % (t, self._max_moment_order))
        binomial = self._binomial_table[:t + 1, :t + 1]
        signs = np.zeros((t + 1, t + 1), dtype=np.float64)
        for i in range(t + 1):
            for j in range(t + 1):
                signs[i, j] = 1.0 - 2 * ((i - j) % 2)
        exponents = np.array([j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                              for j in range(t + 1)], dtype=np.float64)
        x = binomial * signs
        # print("x:",x)
        # print("exponents:",exponents)
        y = x * np.exp(exponents)
        z = np.sum(y, axis=1)
        return z

    def _compute_log_moment(self, sigma, q, moment_order):
        """Compute high moment of privacy loss."""
        assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                        "of the upper bound %d."
                                                        % (moment_order,
                                                           self._max_moment_order))
        binomial_table = self._binomial_table[moment_order, :moment_order + 1]
        qs = np.exp(np.arange(moment_order + 1) * np.log(q))
        moments0 = self._differential_moments(sigma, 0.0, moment_order)
        term0 = np.sum(binomial_table * qs * moments0)
        moments1 = self._differential_moments(sigma, 1.0, moment_order)
        term1 = np.sum(binomial_table * qs * moments1)
        return np.log(q * term0 + (1.0 - q) * term1)