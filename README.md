# RLinf-dexterous-hands

Dexterous hand and data-glove drivers for RLinf.

Includes drivers for:

- **Ruiyan** five-finger dexterous hand (custom serial protocol)
- **Aoyi** five-finger dexterous hand (Modbus RTU)
- **PSI data-glove** (serial + calibration + filtering)

## Installation

```bash
# Base (Ruiyan hand only — needs numpy + pyserial)
pip install RLinf-dexterous-hands

# With glove support (adds pyyaml)
pip install "RLinf-dexterous-hands[glove]"

# With Aoyi hand support (adds pymodbus)
pip install "RLinf-dexterous-hands[aoyi]"

# Everything
pip install "RLinf-dexterous-hands[all]"
```

## Usage

```python
# Ruiyan hand
from rlinf_dexhand.ruiyan import RuiyanHandDriver

hand = RuiyanHandDriver(port="/dev/ttyUSB0")
hand.initialize()
state = hand.get_state()   # np.ndarray (6,), [0, 1]
hand.command(state)
hand.shutdown()

# Aoyi hand
from rlinf_dexhand.aoyi import AoyiHandDriver

hand = AoyiHandDriver(port="/dev/ttyUSB0", node_id=2)
hand.initialize()
state = hand.get_state()
hand.command(state)
hand.shutdown()

# PSI data-glove
from rlinf_dexhand.glove import GloveExpert

expert = GloveExpert(left_port="/dev/ttyACM0", right_port=None)
angles = expert.get_angles()   # np.ndarray (6,), [0, 1]
expert.close()
```
