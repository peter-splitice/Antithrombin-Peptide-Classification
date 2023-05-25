## Imports
import os
import sys

# Set current and parent directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the current path
sys.path.append(parent_dir)

# Operating Ssytem
from peter_initializations import *