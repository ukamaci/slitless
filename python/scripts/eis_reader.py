import matplotlib.pyplot as plt
import eispac
pathdir = '/home/kamo/resources/slitless/'

if __name__ == '__main__':
    # NB: The "name guard" above is important for running safe, parallel fitting
    #     in a stand-alone script. If running in an interactive shell, eispac will
    #     default to a single core. If you _really_ know what you are doing,
    #     you can override it. Just be careful.

    # Select local files (relative paths are fine)
    eis_filepath = pathdir + 'data/eis_20070512_094534.data.h5'
    template_filepath = pathdir + 'data/templates/fe_12_195_119.1c.template.h5'

    # Load the data and fit template
    # Note: read_cube() performs basic pointing corrections and
    #       applies the pre-flight radiometric calibration
    data_cube = eispac.read_cube(eis_filepath, window=195.119)
    tmplt = eispac.read_template(template_filepath)

    # Fit the spectra
    fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu='max', unsafe_mp=True)

    # Save the full fit results and export select measurements to FITS files
    result_files = eispac.save_fit(fit_res, save_dir=pathdir+'data/fits/')
    # FITS_files = eispac.export_fits(fit_res, save_dir='cwd')
