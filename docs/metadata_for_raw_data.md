Metadata Files for Raw Data Files Purpose In order to provide clear
documentation for each raw dataset, we include a metadata file in the
<qpb_main_program> directory for each project (i.e., each dataset). These
metadata files help to explain the context and purpose behind the data
collection process. They serve as an essential reference for understanding the
choices made during data generation, the parameters that were varied, and the
assumptions made.

These metadata files are intended to be self-contained documents that describe
the datasets stored in the corresponding project subdirectories under
data_files/raw. They ensure that anyone working with the data can easily
understand how the dataset was created and what each parameter represents.

File Location For each project, a markdown file named project_name_metadata.md
is placed directly inside the <qpb_main_program> directory. This is done to keep
the data subdirectories, such as <qpb_main_program>/<project_name>, focused only
on raw data files, without cluttering them with additional documentation.

The location of the metadata file in the main program directory makes it easy to
access and ensures that the data and its corresponding documentation are always
paired together.

File Structure Each metadata file should include the following sections:

Purpose of Data Collection A brief description of why the data was collected and
what the goals were. This can include details about the study or experiment that
led to the creation of the dataset.

Fixed Parameters A list of parameters that were kept constant during data
collection. This helps provide context to the varying parameters that are the
focus of the data.

Varying Parameters A list of parameters that were varied during data collection,
including the range of values used and any relevant details about the
variations.

Notes on Data Collection Any additional relevant information about the data
collection process, such as the experimental setup, tools used, or any issues
that might affect the interpretation of the data.

Example Metadata File Here’s an example of the content that might appear in a
project_name_metadata.md file:

markdown Copy code
# Project: ProjectName - Metadata

## Purpose of Data Collection
- This data was collected to analyze the effects of temperature on the growth
  rate of species X. 
- The goal was to observe how varying temperature levels influence the growth
  patterns over a period of time.

## Fixed Parameters
- Species Type: Species X
- Growth Medium: Medium Y

## Varying Parameters
- Temperature: Varying from 20°C to 40°C in 5°C increments
- Light Exposure: Varying from 6 to 12 hours per day

## Notes on Data Collection
- The data was generated under controlled laboratory conditions, using
  standardized equipment.
- There were no known issues with the data collection process, though data from
the 30°C group showed slight inconsistencies due to equipment malfunction.
Benefits Including metadata files with each project helps ensure that:

The data's context and purpose are well-documented, preventing any confusion
when revisiting the dataset later. Collaborators or users of the data can easily
understand the structure of the dataset, including what parameters were
controlled and which ones were varied. Future data analyses will have an
accessible reference for understanding the experimental setup and assumptions
behind the data.