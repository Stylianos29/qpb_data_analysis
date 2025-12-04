# Tutorial: Experiment Logging for QPB Data Analysis

## Introduction

This tutorial explains how to systematically document your computational experiments using structured YAML files. Good experiment logging is essential for:

- **Reproducibility**: Understanding what you did months/years later
- **Collaboration**: Helping others understand your work
- **Scientific rigor**: Maintaining experimental records like a lab notebook
- **Paper writing**: Having all methodology and rationale documented
- **Resource planning**: Tracking computational costs and decisions

## Directory Structure

Experiment metadata files are stored alongside your data:

```
qpb_data_analysis/data_files/raw/
├── invert/
│   ├── Experiment_Name_1/               # Data directory
│   ├── Experiment_Name_1.yaml           # Experiment log ← HERE
│   ├── Experiment_Name_2/
│   └── Experiment_Name_2.yaml
└── sign_squared_violation/
    ├── Experiment_Name_3/
    └── Experiment_Name_3.yaml
```

**Key principle**: Each experiment directory has a corresponding `.yaml` file as its sibling.

## YAML File Structure

### Required Sections

Every experiment log should contain these core sections:

#### 1. **Experiment Identification**
```yaml
experiment:
  id: "unique_experiment_id"
  name: "Descriptive_Name"
  date_created: "YYYY-MM-DD"
  date_completed: "YYYY-MM-DD"
  status: "completed"  # or: planning, running, failed, archived
  researcher: "Your Name"
```

#### 2. **Scientific Context**
```yaml
motivation:
  primary_question: |
    What scientific question are you trying to answer?
  
  background: |
    Why are you doing this? What prompted this experiment?
  
  hypothesis: |
    What do you expect to find?
```

**Why this matters**: Six months from now, you'll forget the "why". This captures your scientific thinking.

#### 3. **Methodology**
```yaml
methodology:
  approach: "Brief description of approach"
  
  key_decisions:
    decision_1_name: |
      Explain why you made this choice.
      What alternatives did you consider?
```

**Example decisions to document**:
- Why these specific parameter values?
- Why this gauge configuration?
- Why this number of spinors?
- Any compromises made due to computational constraints?

#### 4. **Parameters**
```yaml
parameters:
  varied:
    - name: "parameter_name"
      values: [1, 2, 3]
      rationale: "Why these values?"
  
  fixed:
    parameter_name: value
    parameter_name_note: "Why fixed at this value?"
```

**Critical**: Always document the rationale, not just the values.

#### 5. **Results & Next Steps**
```yaml
preliminary_findings:
  summary: "What did you observe?"
  
interpretation:
  main_conclusion: "What does it mean?"
  caveats: ["Limitation 1", "Limitation 2"]

next_steps:
  immediate: ["Action 1", "Action 2"]
  future_experiments: ["If X, then Y"]
```

## Workflow: How to Use This System

### Before Running (Planning Phase)

1. **Create the YAML file**
   ```bash
   cd qpb_data_analysis/data_files/raw/invert/
   touch My_Experiment.yaml
   ```

2. **Fill in core sections** (copy from template)
   - Experiment ID and name
   - Scientific question and motivation
   - Planned parameters
   - Expected outcomes

3. **Status**: Set to `"planning"` or `"running"`

### During Runs

4. **Add runtime notes**
   ```yaml
   runtime_observations:
     - "2024-11-15 14:30: Started runs on JURECA"
     - "2024-11-15 18:00: m=0.02 runs taking longer than expected"
     - "2024-11-16 09:00: All runs completed successfully"
   ```

### After Completion

5. **Update results section**
   - What you actually found
   - How it compares to expectations
   - Preliminary quantitative results

6. **Document next steps**
   - What analysis to do
   - What follow-up experiments needed
   - Decision points

7. **Update status**: Set to `"completed"`

## Common Experiment Types

### Cost Estimation Studies

**Purpose**: Determine computational feasibility before production runs

**Key things to document**:
```yaml
methodology:
  key_decisions:
    resource_optimization: |
      Used reduced spinors/configurations for cost estimation only.
      Explain why this is valid for cost estimates.

data_files:
  file_types:
    - extension: ".dat"
      status: "DISCARDED"
      rationale: "Not physically meaningful - only needed timing data"
```

### Parameter Sweeps

**Purpose**: Explore parameter space to optimize settings

**Key things to document**:
```yaml
parameters:
  varied:
    - name: "sweep_parameter"
      values: [...]
      rationale: "Why this range? Why this granularity?"
      
preliminary_findings:
  optimal_range: "What range looks promising?"
  
next_steps:
  immediate:
    - "Zoom in on range X-Y with finer resolution"
```

### Production Runs

**Purpose**: Generate statistics for publication-quality results

**Key things to document**:
```yaml
scientific_goals:
  publication_target: "Which paper/analysis?"
  required_statistics: "How many configurations needed?"

quality_assurance:
  validation_checks:
    - "Thermalization check: PASSED"
    - "Autocorrelation analysis: PASSED"
```

### Debugging / Troubleshooting

**Purpose**: Investigate unexpected behavior or failures

**Key things to document**:
```yaml
motivation:
  problem_statement: "What went wrong in previous experiment?"
  
methodology:
  troubleshooting_approach: |
    Systematically vary X to isolate the issue.
    Compare with working configuration Y.

findings:
  root_cause: "Problem identified as Z"
  solution: "Fixed by doing W"
```

## Best Practices

### ✅ DO:

1. **Write before you run**: Create the YAML before generating data
2. **Explain your reasoning**: Future-you needs to know why you made choices
3. **Document surprises**: Unexpected results are often the most interesting
4. **Be honest about limitations**: Note compromises, assumptions, caveats
5. **Link related experiments**: Reference previous/related work
6. **Update after analysis**: Add findings once you've analyzed the data

### ❌ DON'T:

1. **Don't wait until paper-writing**: By then, you'll have forgotten key details
2. **Don't just list parameters**: Explain why those values
3. **Don't hide failures**: Failed experiments teach important lessons
4. **Don't over-format**: YAML is meant to be readable, not beautiful
5. **Don't duplicate filename info**: The YAML should add context, not repeat names

## Templates

### Minimal Template (Quick Experiments)

```yaml
experiment:
  id: "quick_test_001"
  date: "2024-11-15"
  status: "completed"

purpose: "One-line description of what you're testing"

parameters:
  varied: {param: [values]}
  fixed: {param: value}

result: "What you found"

next: "What to do next"
```

### Full Template

See the comprehensive example in `templates/experiment_metadata_template.yaml`

## Querying Your Experiment Logs

Your YAML files are machine-readable, so you can programmatically search them:

```python
import yaml
import glob

# Find all experiments from 2024
for yaml_file in glob.glob("data_files/raw/*/*.yaml"):
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    if data['experiment']['date_created'].startswith('2024'):
        print(f"Found: {data['experiment']['name']}")
```

## Integration with Analysis Pipeline

Your analysis scripts can read these YAML files:

```python
def load_experiment_metadata(data_dir):
    """Load experiment metadata from parent directory."""
    yaml_file = data_dir.replace('./', '').rstrip('/') + '.yaml'
    parent_yaml = os.path.join(os.path.dirname(data_dir), yaml_file)
    
    if os.path.exists(parent_yaml):
        with open(parent_yaml) as f:
            return yaml.safe_load(f)
    return None

# Use in analysis
metadata = load_experiment_metadata(raw_data_dir)
if metadata:
    print(f"Analyzing: {metadata['experiment']['name']}")
    print(f"Purpose: {metadata['motivation']['primary_question']}")
```

## FAQ

**Q: How detailed should I be?**  
A: Detailed enough that someone (including future-you) can understand your reasoning. If you're unsure, include it.

**Q: What if I don't know the results yet?**  
A: Leave those sections as "TBD" or "In progress". Update them later.

**Q: Can I modify the template?**  
A: Absolutely! Adapt it to your needs. The key is consistency within your project.

**Q: What about failed experiments?**  
A: Document them! Failed experiments are valuable. Set `status: "failed"` and document what went wrong and what you learned.

**Q: Should I version control these?**  
A: Yes! Commit your YAML files to git along with your data directory structure.

## Summary

Good experiment logging is about capturing the **narrative** of your research:
- What you were trying to learn (motivation)
- How you designed the experiment (methodology)
- What choices you made and why (rationale)
- What you found (results)
- What it means (interpretation)
- What comes next (future work)

Think of it as a lab notebook for computational experiments. Your future self (and your collaborators, and your paper reviewers) will thank you!

## Additional Resources

- Template: `templates/experiment_metadata_template.yaml`
- Examples: `examples/experiment_logs/`
- YAML syntax guide: https://yaml.org/spec/1.2/spec.html
- Project documentation: `docs/`
