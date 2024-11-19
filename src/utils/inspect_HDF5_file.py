import click # type: ignore
import h5py


@click.command()
@click.option("--hdf5_file_path", "hdf5_file_path",
                "-hdf5", default=None,
                help="Path to the HDF5 file to be inspected.")

def main(hdf5_file_path):

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Access the group for the specific filename
        # TODO: Make it more general
        file_group = hdf5_file[
            'KL_Brillouin_mu1p0_rho1p0_cSW0_EpsCG1e-16_config0000200_n1.txt']
        
        # Retrieve and print the stored datasets
        for parameter in file_group:
            print(f"{parameter}: {file_group[parameter][:]}")


if __name__ == "__main__":
    main()
