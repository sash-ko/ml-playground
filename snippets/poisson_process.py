import numpy as np


def poisson_process(time_period, num_events):
    """ Return a list of time points when events happened
    during time_period, e.g. [0.12, 1.23, 1.56, 2.33, 4.76]
    
    NOTE: the number of time points many not be equal to num_events
    """
    lambda_ = num_events / time_period

    events = np.random.exponential(1.0 / lambda_, num_events)
    return np.cumsum(events)


def discrete_poisson_process(duration, num_events):
    """ Poisson process to generate a list of event counts for each discrete
    step of "duration":

    >> poisson_process(10, 8)
    >> [0 1 1 3 1 2 0 0 0 0]
    """

    lambda_ = num_events / duration

    events = np.random.exponential(1.0 / lambda_, num_events)
    events = (np.cumsum(events)).round().astype(int)
    # count size of each bin
    events = np.bincount(events)

    # since "events" can differ from "num_events" adjust the size of the array
    events = np.concatenate((events, [0] * (duration - events.shape[0])))
    return events[:duration].astype(int)
