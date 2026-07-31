# DeeplearningExamples

Examples of deep-learning methods applied to biological time-course data.

## Installation

Install the repository in editable mode before running the notebooks:

```bash
python -m pip install -e .
```

Install the framework required by the notebook:

```bash
python -m pip install -e ".[torch]"
python -m pip install -e ".[tensorflow]"
```

After installation, project modules can be imported without modifying
`sys.path`:

```python
from deeplearning_examples.io import load_smad_timecourses
from deeplearning_examples.models import TimecourseClassifier
```

Editable installation is recommended because the example datasets remain in
the repository under `Data/` and are not copied into the Python package.
