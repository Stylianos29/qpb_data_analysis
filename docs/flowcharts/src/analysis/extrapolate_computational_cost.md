# Computational Cost Extrapolation Process Flow

This flowchart illustrates the process flow for the
`extrapolate_computational_cost.py` script, which automatically detects
and applies the appropriate extrapolation method based on input files
provided.

```mermaid
flowchart TD
    A[Start: extrapolate_computational_cost.py] --> B[Validate Configuration]
    B --> C[Setup Directories & Logging]
    C --> D[Auto-detect Extrapolation Method]
    
    D --> E{PCAC CSV provided?}
    E -->|Yes| F[Method: fixed_pcac_mass]
    E -->|No| G[Method: fixed_bare_mass]
    
    F --> H[Load Reference PCAC Mass<br/>from Configuration]
    G --> I[Load Reference Bare Mass<br/>from Configuration]
    
    H --> J[Execute PCAC-based Analysis]
    I --> K[Execute Direct Analysis]
    
    %% PCAC-based Analysis Branch
    J --> L[Load & Prepare PCAC Data]
    L --> M[Create PCAC DataPlotter]
    M -->     N[Perform Linear Fit:<br/>PCAC_mass vs bare_mass]
    N --> O[Invert Fit to Get Reference Bare Mass:<br/>bare_mass from PCAC reference]
    O --> P[Reference Bare Mass with Uncertainty<br/>gvar object]
    
    %% Direct Analysis Branch  
    K --> Q[Reference Bare Mass<br/>float value]
    
    %% Common Cost Analysis
    P --> R[Load & Prepare Cost Data]
    Q --> R
    R --> S[Average Costs Across Configurations<br/>using DataFrameAnalyzer]
    S --> T[Create Cost DataPlotter]
    T -->     U[Perform Shifted Power Law Fit:<br/>cost vs bare_mass]
    U --> V[Generate Cost Plots with<br/>Extrapolation Lines]
    
    %% Export Phase
    V --> W[Export Results to CSV]
    W --> X[Generate Summary Report]
    X --> Y[End: Success]
    
    %% Error Handling
    B -->|Validation Fails| Z[Exit: Configuration Error]
    L -->|Load Fails| AA[Exit: PCAC Data Error]
    R -->|Load Fails| BB[Exit: Cost Data Error]
    N -->|Fit Fails| CC[Exit: PCAC Fit Error]
    U -->|Fit Fails| DD[Exit: Cost Fit Error]
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef method fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef data fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class A,Y startEnd
    class B,C,L,M,N,O,R,S,T,U,V,W,X process
    class E decision
    class F,G,H,I,J,K method
    class Z,AA,BB,CC,DD error
    class P,Q data
```

## Key Process Components

### 1. **Automatic Method Detection**
- **Input Analysis**: Script examines command-line arguments
- **Method Selection**: 
  - `fixed_pcac_mass` if PCAC CSV provided (`-i_pcac`)
  - `fixed_bare_mass` if only processed CSV provided (`-i_proc`)

### 2. **PCAC-based Analysis Path**
- **PCAC Data Processing**: Load plateau PCAC mass estimates
- **Linear Fitting**: PCAC mass vs bare mass relationship
- **Inversion**: Convert reference PCAC mass to bare mass with
  uncertainty
- **Uncertainty Propagation**: Uses gvar objects for error handling

### 3. **Direct Analysis Path**
- **Configuration**: Uses pre-configured reference bare mass value
- **Direct Processing**: Skips PCAC analysis step

### 4. **Common Cost Analysis**
- **Data Preparation**: Load computational cost data
- **Configuration Averaging**: Average across multiple configurations
- **Curve Fitting**: Shifted power law fit for cost extrapolation
- **Visualization**: Generate plots with extrapolation lines

### 5. **Output Generation**
- **CSV Export**: Extrapolated cost predictions with uncertainties
- **Plot Generation**: Professional visualizations with fit curves
- **Summary Reporting**: Analysis results and statistics

## Input Requirements

| Method | Required Files | Purpose |
|--------|---------------|---------|
| `fixed_bare_mass` | `processed_parameter_values.csv` | Direct cost extrapolation |
| `fixed_pcac_mass` | `processed_parameter_values.csv`<br/>`plateau_PCAC_mass_estimates.csv` | PCAC-mediated extrapolation |

## Output Files

- **CSV**: `computational_cost_extrapolation.csv` (in output directory)
- **Plots**: Cost and PCAC analysis plots (in plots directory)
- **Logs**: Detailed execution logs (if logging enabled)
