```{jupyter-kernel} python3
:id: quantify_core_all_docs
```

```{jupyter-execute}
:hide-code:

# Prettify the outputs in the entire API reference
from rich import pretty
pretty.install()
```

% autodoc does not play nicely with myst-parser, so it must be used within RST context
# quantify_core

```{eval-rst}
.. automodule:: quantify_core
   :members:
```

(analysis-api)=

## analysis

```{eval-rst}
.. automodule:: quantify_core.analysis
    :members:

```

### base_analysis

```{eval-rst}
.. automodule:: quantify_core.analysis.base_analysis
    :members:
    :show-inheritance:
```

### cosine_analysis

```{eval-rst}
.. automodule:: quantify_core.analysis.cosine_analysis
    :members:
    :show-inheritance:
```

### spectroscopy_analysis

```{eval-rst}
.. automodule:: quantify_core.analysis.spectroscopy_analysis
    :members:
    :show-inheritance:
```

### single_qubit_timedomain

```{eval-rst}
.. automodule:: quantify_core.analysis.single_qubit_timedomain
    :members:
    :show-inheritance:

```

### interpolation_analysis

```{eval-rst}
.. automodule:: quantify_core.analysis.interpolation_analysis
    :members:
    :show-inheritance:
```

### optimization_analysis

```{eval-rst}
.. automodule:: quantify_core.analysis.optimization_analysis
    :members:
    :show-inheritance:

```

### fitting_models

```{eval-rst}
.. automodule:: quantify_core.analysis.fitting_models
    :members:
    :show-inheritance:

```

### calibration

```{eval-rst}
.. automodule:: quantify_core.analysis.calibration
    :members:
    :show-inheritance:

```

## data

### types

```{eval-rst}
.. automodule:: quantify_core.data.types
    :members:
```

### handling

```{eval-rst}
.. automodule:: quantify_core.data.handling
    :members:
```

### dataset_adapters

```{eval-rst}
.. automodule:: quantify_core.data.dataset_adapters
    :members:
```

### dataset_attrs

```{eval-rst}
.. automodule:: quantify_core.data.dataset_attrs
    :members:
```

### experiment

```{eval-rst}
.. automodule:: quantify_core.data.experiment
    :members:

```

## measurement

```{eval-rst}
.. automodule:: quantify_core.measurement
    :members:
```

### types

```{eval-rst}
.. automodule:: quantify_core.measurement.types
    :members:
```

### control

```{eval-rst}
.. automodule:: quantify_core.measurement.control
    :members:

```

## utilities

### experiment_helpers

```{eval-rst}
.. automodule:: quantify_core.utilities.experiment_helpers
    :members:
```

### dataset_examples

```{eval-rst}
.. automodule:: quantify_core.utilities.dataset_examples
    :members:
```

### examples_support

```{eval-rst}
.. automodule:: quantify_core.utilities.examples_support
    :members:
```

### deprecation

```{eval-rst}
.. automodule:: quantify_core.utilities.deprecation
   :members:
```

## visualization

```{eval-rst}
.. automodule:: quantify_core.visualization
    :members:
```

### instrument_monitor

```{eval-rst}
.. automodule:: quantify_core.visualization.instrument_monitor
    :members:
```

### pyqt_plotmon

```{eval-rst}
.. automodule:: quantify_core.visualization.pyqt_plotmon
    :members:
```

### color_utilities

```{eval-rst}
.. automodule:: quantify_core.visualization.color_utilities
    :members:
```

### mpl_plotting

```{eval-rst}
.. automodule:: quantify_core.visualization.mpl_plotting
    :members:
```

### plot_interpolation

```{eval-rst}
.. automodule:: quantify_core.visualization.plot_interpolation
    :members:
```

### SI Utilities

```{eval-rst}
.. automodule:: quantify_core.visualization.SI_utilities
    :members:
```

# bibliography

```{eval-rst}
.. bibliography::
```
