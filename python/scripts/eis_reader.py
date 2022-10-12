import matplotlib.pyplot as plt
import eispac
import numpy as np
pathdir = '/home/kamo/resources/slitless/data/eis_data/'

dates = [
#    '20070124_181113',
#    '20070127_034020',
#    '20070128_061012',
#    '20070326_183342',
#    '20070512_094534',
#    '20070702_120742',
#    '20070708_105820',
#    '20070929_140227',
#    '20071004_140220',
#    '20071211_002416',
#    '20080106_105243',
#    '20080203_073210'
   '20091211_195014'
]

if __name__ == '__main__':
    # NB: The "name guard" above is important for running safe, parallel fitting
    #     in a stand-alone script. If running in an interactive shell, eispac will
    #     default to a single core. If you _really_ know what you are doing,
    #     you can override it. Just be careful.

    for date in dates:
        print(date)
        # download file if it doesn't already exist
        eispac.download.download_hdf5_data(
            filename=f'eis_l0_{date}.fits.gz', 
            local_top=pathdir+'l2/')

        # Select local files (relative paths are fine)
        eis_filepath = pathdir + f'l2/eis_{date}.data.h5'
        template_filepath = pathdir + 'templates/fe_12_195_119.1c.template.h5'

        # Load the data and fit template
        # Note: read_cube() performs basic pointing corrections and
        #       applies the pre-flight radiometric calibration
        data_cube = eispac.read_cube(eis_filepath, window=195.119)
        tmplt = eispac.read_template(template_filepath)

        # Fit the spectra
        fit_res = eispac.fit_spectra(data_cube, tmplt, ncpu='max', unsafe_mp=True)

        np.save(pathdir+'dataset/'+f'int_{date}.npy', fit_res.get_map(0, 'int').data)
        np.save(pathdir+'dataset/'+f'vel_{date}.npy', fit_res.get_map(0, 'vel').data)
        np.save(pathdir+'dataset/'+f'width_{date}.npy', fit_res.get_map(0, 'width').data)

        # Save the full fit results and export select measurements to FITS files
        # result_files = eispac.save_fit(fit_res, save_dir=pathdir+'fits/')
        # FITS_files = eispac.export_fits(fit_res, save_dir='cwd')
