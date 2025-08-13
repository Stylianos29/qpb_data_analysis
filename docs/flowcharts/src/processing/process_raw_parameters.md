# Parsing Scripts Flowchart

## Design Goal
Convert raw simulation files into structured, hierarchical HDF5 format
for efficient analysis.

## Flowchart

```mermaid
flowchart TD
    %% START
    Start([Start Parsing Script]) --> ScriptType{Script Type?}
    
    %% SCRIPT SELECTION
    ScriptType -- Log Files --> LogInput[/Input: .txt files directory/]
    ScriptType -- Correlator Files --> CorrInput[/Input: .dat files directory/]
    
    %% LOG FILES PATH
    LogInput --> LogValidation[Validate Input Directory]
    LogValidation --> LogLogging[Initialize Logging]
    LogLogging --> LogFileCheck[Check for .txt files]
    
    LogFileCheck -- No .txt files --> LogEarlyExit[Log: No files found<br/>Exit cleanly]
    LogFileCheck -- .txt files found --> LogProcess[Process Log Files:<br/>Extract parameters from<br/>filenames and contents]
    
    %% CORRELATOR FILES PATH
    CorrInput --> CorrValidation[Validate Input Directory]
    CorrValidation --> CorrLogging[Initialize Logging]
    CorrLogging --> CorrFileCheck[Check for .dat files]
    
    CorrFileCheck -- No .dat files --> CorrEarlyExit[Log: No files found<br/>Exit cleanly]
    CorrFileCheck -- .dat files found --> CorrProcess[Process Correlator Files:<br/>Extract parameters from filenames<br/>Parse correlator arrays]
    
    %% SHARED PROCESSING (from _shared_parsing.py)
    LogProcess --> SharedClassify[_classify_parameters_by_uniqueness:<br/>Analyze parameter variations<br/>across all files]
    CorrProcess --> SharedClassify
    
    SharedClassify --> ParamTypes{Parameter<br/>Classification}
    
    ParamTypes --> ConstantParams[Constant Parameters:<br/>Same value across all files<br/>→ Second-to-deepest level attributes]
    ParamTypes --> MultivaluedParams[Multivalued Parameters:<br/>Different values across files<br/>→ File-level group attributes]
    
    %% HDF5 STRUCTURE CREATION
    ConstantParams --> HDF5Structure[_create_hdf5_structure_with_constant_params:<br/>Create hierarchical HDF5 groups<br/>Add constant params as attributes]
    MultivaluedParams --> HDF5Structure
    
    %% OUTPUT GENERATION
    HDF5Structure --> OutputType{Output Type}
    
    %% LOG FILES OUTPUT
    OutputType -- Log Files --> LogCSV[_export_dataframe_to_csv:<br/>Export scalar parameters<br/>to CSV file]
    LogCSV --> LogHDF5[_export_arrays_to_hdf5_with_proper_structure:<br/>Export array parameters<br/>to HDF5 file]
    
    %% CORRELATOR FILES OUTPUT
    OutputType -- Correlator Files --> CorrHDF5[_export_arrays_to_hdf5_with_proper_structure:<br/>Export correlator arrays<br/>to HDF5 file]
    
    %% COMPLETION
    LogHDF5 --> Success[Log: Processing completed<br/>Terminate logging]
    CorrHDF5 --> Success
    LogEarlyExit --> End
    CorrEarlyExit --> End
    Success --> End([End])
    
    %% STYLING
    classDef inputOutput fill:#e1f5fe
    classDef process fill:#f3e5f5
    classDef decision fill:#fff3e0
    classDef shared fill:#e8f5e8
    classDef exit fill:#ffebee
    
    class LogInput,CorrInput,LogCSV,LogHDF5,CorrHDF5 inputOutput
    class LogProcess,CorrProcess,LogValidation,CorrValidation,LogLogging,CorrLogging process
    class ScriptType,LogFileCheck,CorrFileCheck,ParamTypes,OutputType decision
    class SharedClassify,ConstantParams,MultivaluedParams,HDF5Structure shared
    class LogEarlyExit,CorrEarlyExit exit
```

## Key Components

### Shared Library Functions (_shared_parsing.py)
- **`_classify_parameters_by_uniqueness`**: Separates constant from
  multivalued parameters
- **`_create_hdf5_structure_with_constant_params`**: Builds HDF5
  hierarchy with proper attribute placement
- **`_export_arrays_to_hdf5_with_proper_structure`**: Writes data
  following project structure protocol
- **`_export_dataframe_to_csv`**: Handles CSV export with logging
- **`_check_parameter_mismatches`**: Validates parameter consistency

### Script-Specific Processing
- **Log Files**: Extract from both filenames and file contents, export
  to CSV + HDF5
- **Correlator Files**: Extract from filenames, parse correlator arrays,
  export to HDF5 only

### Early Exit Mechanism
Both scripts now check for relevant files before processing and exit
cleanly if none found.