# Experiment Logging with YAML

Quick guide for documenting your computational experiments.

## Why Log Experiments?

Future-you will forget why you chose specific parameters or what the
original goal was. These YAML files capture the scientific narrative
behind your data.

## File Location

Store YAML files as siblings to experiment directories:

```
data_files/raw/
├── invert/
│   ├── My_Experiment/          ← Data directory
│   ├── My_Experiment.yaml      ← Log file
│   └── ...
```

## Quick Start

1. **Copy the template**:
   ```bash
   cp docs/experiment_logging/experiment_metadata_template.yaml \
      data_files/raw/invert/My_Experiment.yaml
   ```

2. **Fill in 5 sections** (~5 minutes):
   - `date` and `status`
   - `purpose`: Why are you doing this?
   - `method`: What's your approach? Any special decisions?
   - `parameters`: What did you vary? What stayed fixed?
   - `result`: What did you find? (use `[TODO]` if not done yet)
   - `next`: What comes after this?

3. **Done!** You've documented your experiment.

## Minimal Example

```yaml
experiment:
  date: "2024-11-15"
  status: "completed"

purpose: |
  Estimate cost for Zolotarev method to compare with Diagonal KL.

method: |
  Cost study only - used 1 spinor (not 12) to save resources.
  Single config (7200) with large condition number.
  Discarded .dat files - only needed timing from .txt files.

parameters:
  varied:
    ZolOrder: [1, 2, 3, 4, 5]
    m: [0.02, 1.0, 1.4, 1.8]
  
  fixed:
    config: 7200
    NSpinors: 1

result: |
  Cost looks manageable. Not prohibitive for production.
  [TODO: Add iteration counts after analysis]

next: |
  - Analyze logs for detailed costs
  - Extrapolate to 12 spinors
  - Report to supervisors
```

## Key Principles

**DO**:
- Fill it out BEFORE or DURING the run (while it's fresh)
- Explain WHY you made choices, not just WHAT you did
- Use `[TODO]` for sections you'll fill later
- Keep it short - aim for 20-40 lines total

**DON'T**:
- Wait until paper-writing (you'll forget details)
- Just list parameters without rationale
- Over-format or make it perfect
- Skip it because "it's obvious" (it won't be in 6 months)

## Automation

The template uses `{{ placeholders }}` for programmatic filling:

```python
import yaml

template = yaml.safe_load(open('template.yaml'))
template['experiment']['date'] = "2024-11-15"
template['parameters']['varied']['m'] = [0.02, 1.0, 1.4]

with open('My_Experiment.yaml', 'w') as f:
    yaml.dump(template, f)
```

## Reading Logs

Your analysis scripts can read these:

```python
import yaml
import os

def get_experiment_metadata(data_dir):
    yaml_file = os.path.basename(data_dir) + '.yaml'
    yaml_path = os.path.join(os.path.dirname(data_dir), yaml_file)
    
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    return None

# Use it
metadata = get_experiment_metadata('data_files/raw/invert/My_Experiment/')
print(f"Purpose: {metadata['purpose']}")
```

## Status Values

- `planning`: Designing the experiment
- `running`: Currently executing
- `completed`: Finished successfully
- `failed`: Run failed (document what went wrong!)
- `archived`: Old experiment, kept for reference

## Tips

1. **Be honest about compromises**: "Used 1 spinor instead of 12 to save
   time" is valuable info
2. **Document failures**: Failed experiments teach lessons
3. **Link related work**: Mention previous experiments by name
4. **Update after analysis**: Add findings once you've looked at the
   data
5. **Version control**: Commit these YAML files with your code

## Full Tutorial

For more detailed guidance, see:
`docs/tutorials/experiment_logging_tutorial.md`

## Files in This Directory

- `experiment_metadata_template.yaml` - Blank template to copy
- `example_Zolotarev_cost_study.yaml` - Filled example
- `README.md` - This file
