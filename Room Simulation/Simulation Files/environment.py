# Importing packages
#import import_ipynb
from additional_functions import *
import numpy as np
import math
import numpy.random as random
random.seed(123)

# Creating a dataset
buffer = 100

#walls

a118 = [[buffer, 751.1 / 2 + buffer, 735 / 2 + buffer, 751.1 / 2 + buffer],
        [buffer, 751.1 / 2 + buffer, buffer, 751.1 / 2 + (1 * 885.8) / 2 + buffer],
        [735 / 2 + buffer, 751.1 / 2 + buffer, 735 / 2 + buffer, 751.1 / 2 + 100 / 2 + buffer],
        [735 / 2 + buffer, 751.1 / 2 + 100 / 2 + 100 / 2 + buffer, 735 / 2 + buffer, 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + buffer],
        [735 / 2 + buffer, 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + buffer, 735 / 2 + buffer, 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 100 / 2 + buffer],
        [buffer, 751.1 / 2 + 885.8 / 2 + buffer, 735 / 2 + buffer, 751.1 / 2 + 885.8 / 2 + buffer]
        ]

walls = a118