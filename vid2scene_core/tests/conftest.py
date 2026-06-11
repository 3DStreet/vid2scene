import os
import sys

# vid2scene_core modules are flat scripts imported by name (e.g. `import pano_sfm`)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
