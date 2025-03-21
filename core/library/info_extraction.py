import re


def operator_type_extraction(operator_type_label):
    """Extracts pieces of information from the working data file or subdirectory name. It returns the operator type ("Standard" or "Brillouin"), the operator method ("Chebyshev" or "KL" - or empty) and the a operator method "enumeration" (if an operator method was implemented), namely either the number of terms N for "Chebyshev" operator method, or the diagonal KL iteration for "KL"."""

    # Operator type
    if "Standard" in operator_type_label:
        operator_type = "Standard"
    else:
        operator_type = "Brillouin"

    # Operator method
    if "Chebyshev" in operator_type_label:
        operator_method = "Chebyshev"
    elif "KL" in operator_type_label:
        operator_method = "KL"
    else:
        operator_method = ""

    # Operator method enumeration
    operator_method_enumeration = ""
    # Check if operator method string is not empty
    if bool(operator_method):
        match = re.search(r"=(\d+)", operator_type_label)
        if match:
            operator_method_enumeration = int(match.group(1))

    return operator_type, operator_method, operator_method_enumeration


class AnalyzeDataset:

    def __init__(self, dataset_file_path, general_operator_type_info: tuple):

        self.dataset_file_path = dataset_file_path

        self.extract_info_from_filename()

        # Additional safety on whether the dataset operator type info is the same as the one extracted from the subdirectory name
        if general_operator_type_info != self.operator_type_info:
            # TODO: Add action!
            pass

        self.extract_content_from_file()

        self.temporal_direction_lattice_size = len(self.contents_array)

    def extract_info_from_filename(self):

        self.operator_type_info = operator_type_extraction(self.dataset_file_path)

        # bare mass
        match = re.search(r"mb_([-]?\d+\.\d+)", self.dataset_file_path)
        if match:
            self.bare_mass = float(match.group(1))
        elif re.search(r"mb=([-]?\d+\.\d+)", self.dataset_file_path):
            match = re.search(r"mb=([-]?\d+\.\d+)", self.dataset_file_path)
            self.bare_mass = float(match.group(1))
        else:
            match = re.search(r"mb([-]?\d+\.\d+)", self.dataset_file_path)
            self.bare_mass = float(match.group(1))

        # configuration label
        match = re.search(r"(\d+).dat", self.dataset_file_path)
        if match:
            self.configuration_label = match.group(1)
        else:
            match = re.search(r"out_(\d+)_", self.dataset_file_path)
            self.configuration_label = match.group(1)

    def extract_content_from_file(self):

        # Pass the file content in a list as the dictionary value
        self.contents_array = []
        with open(self.dataset_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                self.contents_array.append(float(line.strip("\n")))


def extract_lattice_dimensions(line):

    # Extract the substring containing dimensions
    dimensions_substr = line.split("(Lt, Lz, Ly, Lx) = ")[1].strip()
    # Extract time and spatial dimensions
    time_dim_str, spatial_dim_str, _, _ = dimensions_substr.strip("()").split(",")
    # Assign values to variables
    time_dimension = int(time_dim_str)
    spatial_dimension = int(spatial_dim_str)

    return time_dimension, spatial_dimension


def extract_parallel_geometry(line):

    # Extract the substring containing process dimensions
    process_substr = line.split("Processes = ")[1].strip()
    # Extract X, Y, and Z values
    _, x_str, y_str, z_str = process_substr.strip("()").split(",")
    # Assign values to variables
    x_value = int(x_str)
    y_value = int(y_str)
    z_value = int(z_str)

    return x_value, y_value, z_value


def extract_info_from_gauge_links_filename(line):

    # beta value
    match_beta_value = re.search(r"b(\d+)p(\d+)", line)
    beta_value = None
    if match_beta_value:
        i_str, d_str = match_beta_value.groups()
        beta_value = float(i_str + "." + d_str)

    # Lattice dimensions
    match_lattice_dimensions = re.search(r"L(\d+)T(\d+)", line)
    temporal_lattice_size, spatial_lattice_size = None, None
    if match_lattice_dimensions:
        spatial_lattice_size = int(match_lattice_dimensions.group(1))
        temporal_lattice_size = int(match_lattice_dimensions.group(2))

    # APE smearing
    match_APE_smearing = re.search(r"apeN(\d+)a(\d+)p(\d+)", line)
    APE_smearing_iterations, APE_smearing_alpha = None, None
    if match_APE_smearing:
        APE_smearing_iterations = int(match_APE_smearing.group(1))
        APE_smearing_alpha = float(
            match_APE_smearing.group(2) + "." + match_APE_smearing.group(3)
        )

    # Config label
    match = re.search(r"\.00(\d+)00", line)
    config_label = None
    if match:
        config_label = match.group(1)

    return (
        beta_value,
        spatial_lattice_size,
        temporal_lattice_size,
        APE_smearing_iterations,
        APE_smearing_alpha,
        config_label,
    )


def extract_trailing_number(line):
    """Both integer and float in the form of a string."""

    pattern = r"\b\d+(\.\d+)?\b"

    match = re.search(pattern, line)
    if match:
        return match.group()
    else:
        return None


def extract_trailing_exponential(line):
    """Exponential expressions."""

    pattern = r"\b\d+\.\d+e[+-]\d+\b"

    match = re.search(pattern, line)
    if match:
        return float(match.group())
    else:
        return None


def extract_extreme_eigenvalues(line):
    """"""

    pattern = r"[-+]?\d*\.\d+|\d+"
    matches = re.findall(pattern, line)

    if matches:
        alpha = float(matches[0])
        beta = float(matches[1])

        return alpha, beta
    else:
        return None


def extract_CG_iterations_per_vector_inversion(line):

    pattern = r",\s*iters\s*=\s*(\d+)"

    match = re.search(pattern, line)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_multi_shift_CG_iterations(line):

    pattern = r"After (\d+) iterrations msCG converged,"

    match = re.search(pattern, line)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_time_per_vector_inversion(line):

    # pattern = r"t = (\d+\.\d+) secs"
    # pattern = r", t = (\d+\.\d+)"
    pattern = r"t\s*=\s*(\d+\.\d+)\s*secs"

    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    else:
        return None


class invert_log_file:

    def __init__(self, invert_log_file_full_path):

        self.CG_iterations_list = []
        self.multi_shift_CG_iterations_list = []
        self.time_per_vector_inversion_list = []

        with open(invert_log_file_full_path, "r") as file:
            for line in file:
                if "(Lt, Lz, Ly, Lx) =" in line:
                    self.lattice_dimensions = extract_lattice_dimensions(line)

                if "Processes =" in line:
                    self.parallel_geometry = extract_parallel_geometry(line)

                if "Gauge field (raw_32) =" in line:
                    (
                        self.beta_value,
                        self.spatial_lattice_size,
                        self.temporal_lattice_size,
                        self.APE_smearing_iterations,
                        self.APE_smearing_alpha,
                        self.config_label,
                    ) = extract_info_from_gauge_links_filename(line)

                if "APE alpha = " in line:
                    self.APE_alpha = float(extract_trailing_number(line))

                if "APE iterations =" in line:
                    self.APE_iterations = int(extract_trailing_number(line))

                if "kappa =" in line:
                    self.kappa = float(extract_trailing_number(line))

                if "Clover param =" in line:
                    self.clover_parameter = int(extract_trailing_number(line))

                if "Solver epsilon = " in line:
                    self.CG_epsilon = extract_trailing_exponential(line)

                if "Dslash operator is" in line:
                    if "Standard" in line:
                        self.operator_type = "Standard"
                    else:
                        self.operator_type = "Brillouin"

                if "Plaquette =" in line:
                    self.plaquette = float(extract_trailing_number(line))

                if "Done vector = " in line:
                    self.CG_iterations_list.append(
                        extract_CG_iterations_per_vector_inversion(line)
                    )

                if "iterrations msCG converged, t =" in line:
                    self.multi_shift_CG_iterations_list.append(
                        extract_multi_shift_CG_iterations(line)
                    )

                # if ", t = " in line:
                # if re.search(',\s*t\s*=\s*', line):
                if re.search("iters, CG converged, res = ", line):
                    self.time_per_vector_inversion_list.append(
                        extract_time_per_vector_inversion(line)
                    )

                if "CG done, 12 vectors in t = ":
                    self.total_time = extract_trailing_number(
                        line.replace("CG done, 12 vectors", "")
                    )


class Chebyshev_invert_log_file(invert_log_file):

    def __init__(self, invert_log_file_full_path):
        super().__init__(invert_log_file_full_path)

        with open(invert_log_file_full_path, "r") as file:
            for line in file:
                if "Number of Chebyshev polynomial terms =" in line:
                    self.number_of_terms = int(extract_trailing_number(line))

                elif ("alpha = " in line) and ("beta = " in line):
                    self.extreme_eigenvalues_tuple = extract_extreme_eigenvalues(line)


class KL_invert_log_file(invert_log_file):

    def __init__(self, invert_log_file_full_path):
        super().__init__(invert_log_file_full_path)

        with open(invert_log_file_full_path, "r") as file:
            for line in file:
                if "KL class =" in line:
                    self.KL_class = int(extract_trailing_number(line))

                elif "KL iters =" in line:
                    self.KL_iterations = int(extract_trailing_number(line))

                elif "Mu" in line:
                    self.scaling_factor = int(extract_trailing_number(line))
